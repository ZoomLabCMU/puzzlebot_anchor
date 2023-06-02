import casadi as ca
import numpy as np

from puzzlebot_assembly.utils import *

class ControlParam:
    def __init__(self, vmax=0.5,
                    uvmax=0.1,
                    wmax=1.0,
                    uwmax=0.5,
                    gamma=0.1,
                    mpc_horizon=10,
                    constr_horizon=10,
                    eth=1e-3):
        self.vmax = vmax
        self.uvmax = uvmax
        self.wmax = wmax
        self.uwmax = uwmax
        self.gamma = gamma
        self.hmpc = mpc_horizon
        self.hcst = constr_horizon
        self.eth = eth
        self.cost_Q = {
            "cp_xy": 1e5, "cp_t":1e2,      # final cost of connection pair
            "prev_xy": 1e-2, "prev_t": 1e-2,# final cost of connected cp
            #  "s_cp_xy": 1e2, "s_cp_t": 1,  # stage cost of connection pair
            "s_cp_xy": 1e0, "s_cp_t": 1e-2,  # stage cost of connection pair
            "s_prev_xy": 1e-2, "s_prev_t": 1e-3,    # stage cost of conncted cp
            "stay_xyt": 1e-5, "stay_vw": 1e-5,  # initialize cost with staying at the same position
            "zero_xyt": 1e3, # zero out the masked ids
            "smooth_v": 0, "smooth_w":0,
            "Q_u": 1e-1 
            }

class CasadiInterface:
    def __init__(self, N, dt, state_len, M=0.1):
        self.N = N
        self.dt = dt
        self.state_len = state_len
        self.M = M

    def get_local_pt(self):
        xi = ca.SX.sym('xi', 3)
        cp = ca.SX.sym('cp', 2)
        theta = xi[2]
        cp_x = ca.cos(theta)*cp[0] - ca.sin(theta)*cp[1] + xi[0]
        cp_y = ca.sin(theta)*cp[0] + ca.cos(theta)*cp[1] + xi[1]

        return ca.Function("get_local_pt", [xi, cp], [cp_x, cp_y])

    def fk_opt(self, N, dt):
        x_sym = ca.SX.sym('x', 5*N)
        u_sym = ca.SX.sym('u', 2*N)

        theta = x_sym[2::5]
        vs = x_sym[3::5]
        x_dot = ca.SX.zeros(5*N)
        x_dot[0::5] = vs * ca.cos(theta)
        x_dot[1::5] = vs * ca.sin(theta)
        x_dot[2::5] = x_sym[4::5]
        x_dot[3::5] = u_sym[0::2]
        x_dot[4::5] = u_sym[1::2]

        return ca.Function("fk_opt", [x_sym, u_sym], [x_sym + (x_dot * dt)])

    def dd_fx_opt(self, theta, N):
        F = ca.SX.zeros(3*N, 2*N)
        F[0::3, 0::2] = ca.diag(ca.cos(theta))
        F[1::3, 0::2] = ca.diag(ca.sin(theta))
        F[2::3, 1::2] = ca.SX.eye(N)
        return F

class Controller:
    def __init__(self, N, dt, control_param):
        self.N = N
        self.dt = dt
        self.param = control_param
        self.state_len = 5
        self.ca_int = CasadiInterface(N, dt, self.state_len, M=0.1)
        self.fk_opt = self.ca_int.fk_opt(N, dt)
        self.get_local_pt = self.ca_int.get_local_pt()
        #  self.fk_opt = fk_rk4_opt(N, dt)
        #  self.fk_opt = fk_exact_opt(N, dt)
        self.ipopt_param = {"verbose": False, 
                            "ipopt.print_level": 0,
                            "print_time": 0,
                            'ipopt.sb': 'yes',
                            }
        self.opt = None
        self.x = None # 5 states [x, y, theta, v, w, ]
        self.u = None # 2 controls [uv, uw]

        # for debug
        self.prev_x = None
    
    def fit_prev_x2opt(self, prev_x):
        x_curr = np.zeros([self.N, self.state_len])
        x_curr[:, 0:3] = prev_x.reshape([self.N, 3])
        x_curr = x_curr.flatten()
        return x_curr

    def fk(self, x, u):
        '''
        Forward Kinematics
        x: 3N vector, u: 2N vector of velocity
        '''
        x_dot = dd_fx(x[2::3]).dot(u)
        return x + (x_dot * self.dt)

    def init_opt(self, prev_x, prev_u, prev_cp=[]):
        N = self.N
        param = self.param
        opt = ca.Opti()
        sl = self.state_len
        x = opt.variable(sl*N, param.hmpc + 1)
        u = opt.variable(2*N, param.hmpc)

        # for debug
        self.prev_x = prev_x

        # initial state constraints
        opt.subject_to(x[0::sl, 0] == prev_x[0::3])
        opt.subject_to(x[1::sl, 0] == prev_x[1::3])
        opt.subject_to(x[2::sl, 0] == prev_x[2::3])
        opt.subject_to(x[3::sl, 0] == prev_u[0::2])
        opt.subject_to(x[4::sl, 0] == prev_u[1::2])

        # v, w constraints
        opt.subject_to(opt.bounded(-param.vmax, 
                            ca.vec(x[3::sl, :]), param.vmax))
        opt.subject_to(opt.bounded(-param.wmax, 
                            ca.vec(x[4::sl, :]), param.wmax))

        # uv, uw constraints
        opt.subject_to(opt.bounded(-param.uvmax, 
                            ca.vec(u[0::2, :]), param.uvmax))
        opt.subject_to(opt.bounded(-param.uwmax, 
                            ca.vec(u[1::2, :]), param.uwmax))

        #try warm start
        x_curr = self.fit_prev_x2opt(prev_x)
        for ti in range(param.hmpc + 1):
            opt.set_initial(x[:, ti], x_curr)
        opt.set_initial(u, 0)

        self.opt = opt
        self.x = x
        self.u = u

    def add_dynamics_constr(self):
        opt = self.opt
        x, u = self.x, self.u
        # dynamics constraints
        for ti in range(self.param.hmpc):
            opt.subject_to(x[:, ti+1] == self.fk_opt(x[:, ti], u[:, ti]))

    def add_vwlim_constraint(self):
        opt = self.opt
        param = self.param
        sl = self.state_len
        x, u = self.x, self.u

        # try butterfly shape constraints
        for ti in range(self.param.hcst):
            opt.subject_to(-1/param.vmax * ca.fabs(x[3::sl, ti+1]) + 
                        1/param.wmax * x[4::sl, ti+1] <= 0)


    def add_align_poly_constr(self, prev_cp, L):
        if len(prev_cp) == 0: return
        opt = self.opt
        param = self.param
        sl = self.state_len
        get_local_pt = self.get_local_pt
        x, u = self.x, self.u
        
        for cp_ids in prev_cp:
            cp_d = prev_cp[cp_ids][0:2, :]
            
            body_idx = np.where(cp_d[0, :] == L/2)[0] 
            assert(len(body_idx) > 0)
            body_idx = body_idx[0]
            body_id = cp_ids[body_idx]
            anchor_idx = 1 - body_idx
            anchor_id = cp_ids[anchor_idx]
            
            for ti in range(1, param.hcst):
                x_pt = get_local_pt(x[sl*anchor_id:(sl*anchor_id+3), ti], 
                                        ca.MX(cp_d[:, anchor_idx]))
                xr = get_local_pt(x[sl*body_id:(sl*body_id+3), ti],
                                        ca.MX([L/2, -L/2]))
                xl = get_local_pt(x[sl*body_id:(sl*body_id+3), ti],
                                        ca.MX([L/2, L/2]))
                xR = x[sl*body_id, ti]
                yR = x[sl*body_id+1, ti]

                opt.subject_to((x_pt[1] - yR)*(xr[0] - xR) >= (xr[1] - yR)*(x_pt[0] - xR))
                opt.subject_to((x_pt[1] - yR)*(xR - xl[0]) >= (yR - xl[1])*(x_pt[0] - xR))
                opt.subject_to((x_pt[1] - xr[1])*(xl[0] - xr[0]) >= (xl[1] - xr[1])*(x_pt[0] - xr[0]))

    def add_cp_cost(self, cp, ti, xy_param, t_param):
        x, u = self.x, self.u
        assert(self.state_len == 5)
        cost = 0
        i0, i1 = next(iter(cp))
        d0 = cp[(i0, i1)][:, 0]
        d1 = cp[(i0, i1)][:, 1]
        t0 = x[i0*5+2, ti]
        t1 = x[i1*5+2, ti]
        cp_len = d0.shape[0]
        x_diff = x[i0*5:(i0*5+cp_len), -1] - x[i1*5:(i1*5+cp_len), -1]
        x_diff[0] += (ca.cos(t0)*d0[0] - ca.sin(t0)*d0[1] 
                    - (ca.cos(t1)*d1[0] - ca.sin(t1)*d1[1]))
        x_diff[1] += (ca.sin(t0)*d0[0] + ca.cos(t0)*d0[1] 
                    - (ca.sin(t1)*d1[0] + ca.cos(t1)*d1[1]))
        x_diff *= xy_param
        if cp_len > 2:
            # wrap angle diff in tan(0.5x)
            x_diff[2] = ca.tan(0.5*((t0 - t1) - (d0[2] - d1[2])))
            x_diff[2] *= t_param
        cost += ca.mtimes(x_diff.T, x_diff)
        return cost

    def align_cp_cost(self, cp, prev_cp):
        '''
        cp key: (i0, i1)
        cp item: 2-by-2: [[dx0, dy0], [dx1, dy1]].T
        '''
        param = self.param
        cost = 0
        for ti in range(param.hmpc+1):
            for key in cp:
                curr = {key: cp[key]}
                if ti < param.hmpc:
                    cost += self.add_cp_cost(curr, ti, 
                                        param.cost_Q["s_cp_xy"], 
                                        param.cost_Q["s_cp_t"])
                else:
                    cost += self.add_cp_cost(curr, ti,
                                        param.cost_Q["cp_xy"],
                                        param.cost_Q["cp_t"])
            for key in prev_cp:
                curr = {key: prev_cp[key]}
                if ti < param.hmpc:
                    cost += self.add_cp_cost(curr, ti, 
                                        param.cost_Q["s_prev_xy"], 
                                        param.cost_Q["s_prev_t"])
                else:
                    cost += self.add_cp_cost(curr, ti,
                                        param.cost_Q["prev_xy"],
                                        param.cost_Q["prev_t"])
        return cost
    
    def init_cost(self, prev_x, zero_list=[]):
        param = self.param
        sl = self.state_len
        x = self.x
        x_curr = self.fit_prev_x2opt(prev_x)
            
        cost = 0

        # mask zeros for zero_list
        if len(zero_list) > 0:
            vs = ca.vec(x[sl*(zero_list)+3, :])
            ws = ca.vec(x[sl*(zero_list)+4, :])
            cost += ca.mtimes(vs.T, vs) * param.cost_Q["zero_xyt"]
            cost += ca.mtimes(ws.T, ws) * param.cost_Q["zero_xyt"]

        return cost

    def goal_cost(self, goal):
        """
        goal: list of len 3 [x, y, theta]
        """
        assert(self.state_len == 5)
        x, u = self.x, self.u
        goal_vec = np.hstack([goal+[0, 0] for i in range(self.N)])
        x_diff = x[:, -1] - goal_vec
        cost = ca.mtimes(x_diff.T, x_diff)
        return cost

    def stage_cost(self):
        u = self.u
        param = self.param
        cost = 0
        for ti in range(1, param.hmpc):
            cost += ca.mtimes(u[:, ti].T, u[:, ti]) * param.cost_Q["Q_u"]
        return cost

    def smooth_cost(self, prev_u):
        u = self.u
        param = self.param
        diff_u = u[:, 0] - prev_u
        diff_u[0::2] *= param.cost_Q["smooth_v"]
        diff_u[1::2] *= param.cost_Q["smooth_w"]
        cost = ca.mtimes(diff_u.T, diff_u)
        for ti in range(1, param.hmpc):
            diff_u = u[:, ti] - u[:, ti-1]
            diff_u[0::2] *= param.cost_Q["smooth_v"]
            diff_u[1::2] *= param.cost_Q["smooth_w"]
            cost += ca.mtimes(diff_u.T, diff_u)
        return cost

    def optimize_cp(self, cost):
        opt = self.opt
        sl = self.state_len
        opt.minimize(cost)
        opt.solver("ipopt", self.ipopt_param)
        try:
            ans = opt.solve()
            uv = ans.value(self.x[3::sl, 1])
            uw = ans.value(self.x[4::sl, 1])
            return np.vstack([uv, uw]).T.flatten(), ans.value(cost)
        except Exception as e:
            print(e)
            #  print("Solver value: ", opt.debug.value)
            #  opt.debug.show_infeasibilities()
        return np.zeros(2*self.N), None