import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import sys
import argparse
import numpy as np
import pickle as pk
from copy import deepcopy
import pandas as pd
from scipy import stats as st
from scipy.optimize import curve_fit
import itertools
from matplotlib.colors import LinearSegmentedColormap

sns.set_style('ticks', {"axes.linewidth": "1", 'axes.yaxis.grid': False})
basic_cols = ['#ff00aa', '#220022', '#000000', '#002222', '#00aaff']
my_cmap = LinearSegmentedColormap.from_list('mycmap', basic_cols)

default_x_vars = ['constant', 'signal_signed', 'stim', 'stim_x_signal']
default_y_var = 'choice_in'

data_fn_default = '/om/user/rishir/data/optoled/data_yolo_protocol-6.pkl'

parser = argparse.ArgumentParser()
parser.add_argument('--data_fn', default=data_fn_default, type=str)
parser.add_argument('--stim_oi', default=-1, type=int)
parser.add_argument('--n_x_vars', default=4, type=int)
parser.add_argument('--fn_ver', default=1, type=int)
parser.add_argument('--win_size_theta', default=np.pi / 8, type=float)
parser.add_argument('--win_size_rho', default=2, type=float)

parser.add_argument('--include_ignore', default=1, type=int)


class PsychometricFit(object):
    def __init__(self, **kwargs):

        self.n_x_vars = kwargs.get('n_x_vars', 4)
        self.x_vars = default_x_vars[:self.n_x_vars]
        self.y_var = kwargs.get('y_var', default_y_var)
        self.fn_ver = kwargs.get('fn_ver', 1)

        if self.fn_ver == 0:
            self.params = ['logit_choice', 'logit_choice_sessnorm', 'abs_logit_choice', 'abs_logit_choice_sessnorm']
            self.n_params = 4
        elif self.fn_ver == 1:
            self.n_params = self.n_x_vars + 2
            self.params = ['lapse_low', 'lapse_high'] + self.x_vars
        elif self.fn_ver == 2:
            self.n_params = self.n_x_vars + 1
            self.params = ['lapse'] + self.x_vars
        else:
            self.n_params = self.n_x_vars
            self.params = self.x_vars

        self.niter = 5
        self.n_shuf = 10

        return

    @staticmethod
    def psych_1(mu, *params):
        """ 2 params for lapse """
        params_lapse = list(params[:2])
        params_exparg = list(params[2:])

        exparg = 0
        for ig, gv in enumerate(mu.keys()):
            exparg += mu[gv] * params_exparg[ig]
        psychout = (np.exp(-exparg) + 1.0) ** (-1)
        l0, l1 = params_lapse[0], params_lapse[1] - params_lapse[0]
        psychout = l0 + l1 * psychout
        return psychout

    @staticmethod
    def psych_2(mu, *params):
        """ 1 param for lapse """
        params_lapse = list(params[:1])
        params_exparg = list(params[1:])

        exparg = 0
        for ig, gv in enumerate(mu.keys()):
            exparg += mu[gv] * params_exparg[ig]
        psychout = (np.exp(-exparg) + 1.0) ** (-1)
        psychout = params_lapse[0] + (1 - params_lapse[0]) * psychout
        return psychout

    @staticmethod
    def psych_3(mu, *params):
        """ 0 params for lapse """
        params_exparg = list(params[:])

        exparg = 0
        for ig, gv in enumerate(mu.keys()):
            exparg += mu[gv] * params_exparg[ig]
        psychout = (np.exp(-exparg) + 1.0) ** (-1)
        return psychout

    @staticmethod
    def psych_4(mu, *params):
        """ 0 params for lapse """
        params_exparg = list(params[:])

        exparg = 0
        for ig, gv in enumerate(mu.keys()):
            exparg += mu[gv] * params_exparg[ig]
        psychout = (np.exp(-exparg) + 1.0) ** (-1)
        return psychout

    def get_param_estimates_bounds(self, df):
        def get_initial_estimate(df_tmp):
            def logit(y):
                y[y == 0] = 0.01
                y[y == 1] = 0.99
                return np.log(y / (1 - y))

            X = df_tmp.groupby(['signal_signed']).mean()['choice_in'].reset_index()
            logY = logit(X['choice_in'])
            return np.polyfit(X['signal_signed'], logY, 1)

        # these are the parameters for the exponential argument (betas).
        p_lin = get_initial_estimate(df)
        n_x_vars = len(self.x_vars)

        p_0 = np.zeros((4,))
        p_lower = np.zeros((4,))
        p_upper = np.zeros((4,))
        p_0[0], p_lower[0], p_upper[0] = p_lin[1], -10, 10 # constant shift
        p_0[1], p_lower[1], p_upper[1] = p_lin[0], -20, 20 # signal weight (d', or slope)
        p_0[2], p_lower[2], p_upper[2] = 0, -100, 100 # stim weight
        p_0[3], p_lower[3], p_upper[3] = 0, -100, 100 # stim x signal weight

        p_0 = p_0[:n_x_vars]
        p_lower = p_lower[:n_x_vars]
        p_upper = p_upper[:n_x_vars]

        # these are the parameters for the lapses (lambdas).
        yy = np.array(df.groupby('signal_signed')['choice_in'].mean())
        q_0 = np.zeros((2,))
        q_lower = np.zeros((2,))
        q_upper = np.zeros((2,))
        q_0[0], q_lower[0], q_upper[0] = yy.min(), -0.001, 1.001
        q_0[1], q_lower[1], q_upper[1] = yy.max(), -0.001, 1.001

        if self.fn_ver == 1:
            p_0 = np.concatenate((q_0, p_0), axis=0)
            p_lower = np.concatenate((q_lower, p_lower), axis=0)
            p_upper = np.concatenate((q_upper, p_upper), axis=0)
        elif self.fn_ver == 2:
            p_0 = np.concatenate((q_0[:1], p_0), axis=0)
            p_lower = np.concatenate((q_lower[:1], p_lower), axis=0)
            p_upper = np.concatenate((q_upper[:1], p_upper), axis=0)

        return p_0, p_lower, p_upper

    def fit_psych_curve(self, df, df_test):

        if df is None:
            return None

        psych_fns = [self.psych_1, self.psych_2, self.psych_3]
        func = psych_fns[self.fn_ver - 1]
        p_0, p_lower, p_upper = self.get_param_estimates_bounds(df)

        def fit_psych_curve_base(df_):
            popt_, pcov_ = curve_fit(func, df_[self.x_vars], df_[self.y_var],
                                     method='trf', p0=p_0,
                                     bounds=(p_lower, p_upper), maxfev=1000000)
            perr_ = np.sqrt(np.diag(pcov_))
            pred = func(df_[self.x_vars], *popt_)
            # using sum of square error here for ease of comparing across different funcs
            mse_ = np.sum((pred - df_[self.y_var]) ** 2)
            return popt_, perr_, mse_

        def fit_psych_curve_loop(df_):
            opts, opts_err, mses = [], [], []
            for i in range(self.niter):
                popt_, perr_, mse_ = fit_psych_curve_base(df_)
                opts.append(popt_)
                opts_err.append(perr_)
                mses.append(mse_)
            mi = int(np.argmin(mses))
            return opts[mi], opts_err[mi], mses[mi]

        opt, perr, mse = fit_psych_curve_loop(df)
        df_test['psychout'] = func(df_test[self.x_vars], *opt)

        res = {
            'opt': opt,
            'opts_err': perr,
            'mse': mse,
            'pred': df_test,
        }

        return res

    def get_psych_fit_results(self, df):
        def get_test_df():
            choices = {
                'constant': [1],
                'signal_signed': np.linspace(-1, 1, 100),
                'stim': [0, 1],
            }
            choice_keys = choices.keys()
            all_choices = list(itertools.product(*tuple(choices.values())))
            tmp = []
            for model_choice in all_choices:
                x = {}
                for fki, fk in enumerate(choice_keys):
                    x[fk] = model_choice[fki]
                tmp.append(x)
            return pd.DataFrame(tmp)

        df_test = get_test_df()
        res = {}

        if self.n_x_vars == 3:
            res = self.fit_psych_curve(df, df_test)
            if res is not None:
                res['del'] = res['opt'][-1]
                res['z'] = res['opt'][-1] / res['opts_err'][-1]
                res['p'] = st.norm.sf(abs(res['z']))  # one-sided
        elif self.n_x_vars == 2:
            res0 = self.fit_psych_curve(df.query('stim==0'), df_test.query('stim==0'))
            res1 = self.fit_psych_curve(df.query('stim!=0'), df_test.query('stim!=0'))
            res = None
            if (res0 is not None) and (res1 is not None):
                mu_diff = res1['opt'] - res0['opt']
                pooled_var = res0['opts_err'] ** 2 + res1['opts_err'] ** 2
                res = {'res0': res0, 'res1': res1}
                res['del'] = mu_diff[-2]
                res['z'] = mu_diff[-2] / (pooled_var[-2] ** 0.5)
                res['p'] = st.norm.sf(abs(res['z']))  # one-sided
                res['mse'] = res0['mse'] + res1['mse']

        return res


class OptoledAnalyzer(object):
    def __init__(self, **kwargs):
        self.data_fn = kwargs.get('data_fn', data_fn_default)
        self.stim_oi = kwargs.get('stim_oi', 71)  # -1 means include all stim conditions
        self.include_ignore = kwargs.get('include_ignore', 1)

        self.n_x_vars = kwargs.get('n_x_vars', 2)
        self.fn_ver = kwargs.get('fn_ver', 1)
        self.win_size_theta = kwargs.get('win_size_theta', np.pi / 8)

        self.win_size_rho = kwargs.get('win_size_rho', 2)
        self.all_results = None
        self.prefix = self.data_fn.split('/')[-1].split('.')[-2]
        self.data_in = None
        self.load_data()

        self.outfn = '%s_f%d_nvar%d_led%d_win%2.2fx%2.2f' % (self.prefix, self.fn_ver,
                                                             self.n_x_vars, self.stim_oi,
                                                             self.win_size_theta, self.win_size_rho)
        if self.include_ignore == 0:
            self.outfn = '%s_%s' % (self.outfn, 'noIgn')

        self.dat_outdir = '/om/user/rishir/lib/optoled_python/dat/'
        self.fig_outdir = '/om/user/rishir/lib/optoled_python/fig/'

        return

    def load_data(self):
        print('loading %s ' % self.data_fn)
        data_in = pd.read_pickle(self.data_fn)
        if self.stim_oi == -1:
            df_ = data_in
        else:
            query_str = 'stim_led == 0 | stim_led == %d' % self.stim_oi
            df_ = data_in.query(query_str).reset_index(drop=True)
        self.data_in = df_
        return

    @staticmethod
    def select_roi_base(df_in, t_s, t_d):
        def add_new_column(ref_df, col_name, col_vals, col_val_idx):
            x = pd.Series(index=ref_df.index)
            for cii, ci in enumerate(col_val_idx):
                x[ci] = col_vals[cii]
            ref_df[col_name] = x
            return ref_df

        DF = deepcopy(df_in)
        DF = add_new_column(DF, 'in_roi', [1, 0], [t_s, t_d])
        scale_signals = 1.0 if np.nanmax(np.abs(DF['signal_in'])) <= 1 else 100.0
        DF['signal_in'] = DF['signal_in'] / scale_signals
        DF['signal_out'] = DF['signal_out'] / scale_signals
        DF['signal1d'] = DF['signal1d'] / scale_signals

        DF = add_new_column(DF, 'choice_in', [DF['succ'][t_s], DF['fail'][t_d]], [t_s, t_d])
        # ignore trials don't count as choice in or out.
        t_ign = DF['ignore'] == 1
        DF['choice_in'][t_ign] = 0.5
        DF = add_new_column(DF, 'signal_in_roi', [DF['signal_in'][t_s], DF['signal_out'][t_d]], [t_s, t_d])
        DF = add_new_column(DF, 'signal_out_roi', [DF['signal_out'][t_s], DF['signal_in'][t_d]], [t_s, t_d])
        DF = add_new_column(DF, 'signal_signed', [DF['signal1d'][t_s], -1 * DF['signal1d'][t_d]], [t_s, t_d])
        DF = add_new_column(DF, 'constant', [1, 1], [t_s, t_d])

        DF['stim_x_signal'] = DF['stim'] * DF['signal_signed']

        t = t_s | t_d
        df = DF[t].reset_index(drop=True)
        return df

    def select_roi_slice(self, df_in, win_theta_center=0, win_rho_center=5):
        def in_slice(x, y):
            theta = np.mod(np.arctan2(y, x), 2 * np.pi)
            theta_c = np.mod(win_theta_center, 2 * np.pi)
            diff = np.mod(np.abs(theta - theta_c), 2 * np.pi)
            cond1 = diff <= self.win_size_theta

            r = (x ** 2 + y ** 2) ** 0.5
            cond2 = np.abs(r - win_rho_center) < self.win_size_rho
            return cond1 & cond2

        t_s = in_slice(df_in['x'], df_in['y'])
        t_d = in_slice(df_in['xd'], df_in['yd'])
        return self.select_roi_base(df_in, t_s, t_d)

    def get_psych_fit_results_wrapper(self, df):
        angles = np.linspace(-np.pi, np.pi, 100)
        rho_min = 3# + self.win_size_rho
        rho_max = 10# - self.win_size_rho
        rho = list(np.arange(rho_min, rho_max+1))
        all_results = []

        for ang in angles:
            for r in rho:
                df_ = self.select_roi_slice(df, win_theta_center=ang, win_rho_center=r)
                pf = PsychometricFit(n_x_vars=self.n_x_vars, fn_ver=self.fn_ver)
                res = pf.get_psych_fit_results(df_)
                if res is not None:
                    res['theta'] = ang
                    res['r'] = r
                    all_results.append(res)

        self.all_results = all_results
        save_fn = '%s/%s.pkl' % (self.dat_outdir, self.outfn)
        with open(save_fn, 'wb') as f:
            f.write(pk.dumps(all_results))
        return

    def plot_results(self, df):

        def plot_psych_curve(win_index_, ax_):
            res = self.all_results[win_index_]
            ang, r = res['theta'], self.all_results[win_index_]['r']
            df_ = self.select_roi_slice(df, win_theta_center=ang, win_rho_center=r)
            g = df_.groupby(['signal_signed', 'stim'])['choice_in']
            g.mean().unstack().plot(linestyle='--', linewidth=0,
                                    elinewidth=1.5, marker='o',
                                    yerr=g.sem().unstack(), legend=False,
                                    xlim=[-0.5, 0.5], ylim=[0, 1], ax=ax_)
            if 'pred' in res.keys():
                g = res['pred'].groupby(['signal_signed', 'stim'])['psychout']
                g.mean().unstack().plot(linestyle='--', ax=ax_, linewidth=0.5, legend=False)
            else:
                for fk in ['res0', 'res1']:
                    g = res[fk]['pred'].groupby('signal_signed')['psychout']
                    g.mean().plot(linestyle='--', ax=ax_, linewidth=0.5, legend=False)

            ax_.set_xlim([-0.5, 0.5])
            ax_.set_ylim([0, 1])
            ax_.set_aspect(1.0)
            sns.despine(ax=ax_, trim=False, offset=10)
            plt.tight_layout()
            return

        def plot_z_ring(f_, ax_, plot_var='z', masked=False):
            max_r = 10
            z = np.array([r[plot_var] for r in self.all_results])
            mask_z = np.array([r['z'] for r in self.all_results])
            if masked:
                z[np.abs(mask_z) < 1.7] = np.nan

            norm = mpl.colors.Normalize(-3, 3)
            if plot_var != 'z':
                norm = mpl.colors.Normalize(np.nanmin(z), np.nanmax(z))
            theta = np.array([r['theta'] for r in self.all_results])
            radius = np.array([r['r'] for r in self.all_results])
            urad = np.unique(radius)
            for ur in urad:
                t_ur = radius == ur
                t = theta[t_ur]  # theta values
                r = np.linspace(ur, max_r, 2)  # radius values change 0.6 to 0 for full circle
                c = np.tile(z[t_ur], (2, 1))  # define color values as theta value
                im = ax_.pcolormesh(t, r, c, norm=norm, cmap=my_cmap)  # plot the colormesh on axis with colormap
            ax_.set_yticklabels([])  # turn of radial tick labels (yticks)
            f_.colorbar(im, ax=ax_, use_gridspec=True)
            return

        f = plt.figure(figsize=(12, 3))
        ax1 = plt.subplot(141, projection='polar')
        plot_z_ring(f, ax1, plot_var='z')

        ax2 = plt.subplot(142, projection='polar')
        plot_z_ring(f, ax2, plot_var='del', masked=False)

        ax3 = plt.subplot(143, projection='polar')
        plot_z_ring(f, ax3, plot_var='del', masked=True)

        ax4 = plt.subplot(144)
        win_index = np.argmin([r['z'] for r in self.all_results])
        plot_psych_curve(win_index, ax4)
        p_curr = self.all_results[win_index]['p']
        mse_curr = self.all_results[win_index]['mse']
        ax4.set_title('mse=%2.2f \n p=%2.2e' % (mse_curr, p_curr))
        plt.tight_layout()

        f.savefig('%s/%s.pdf' % (self.fig_outdir, self.outfn))
        return

    def run_all(self):
        self.get_psych_fit_results_wrapper(self.data_in)
        self.plot_results(self.data_in)
        return


def main(argv):
    flags_, _ = parser.parse_known_args(argv)
    flags = vars(flags_)
    psych = OptoledAnalyzer(**flags)
    psych.run_all()
    return


if __name__ == "__main__":
    main(sys.argv[1:])
