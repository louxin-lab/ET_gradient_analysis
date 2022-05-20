from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

plt.ioff()

def correlation(x, y, show=False, save=False,
                out_path=None, 
                fig_width=5, fig_height=5,
                x_label='', y_label='',
                x_lim=None, y_lim=None,
                fontsize=14,
                color="#e377c2"):
    r = pearsonr(x, y)[0]
    p = pearsonr(x, y)[1]
    if show or save:
        _, ax = plt.subplots(figsize=(float(fig_width), float(fig_height)))
        ax = sns.regplot(x=x,y=y, robust=True,
                            ax=ax,color=color)
        ax.set_title('r={:.2f}, p={:.2e}'.format(r, p), fontdict={'fontsize': fontsize})
        ax.set_xlabel(x_label, fontsize=fontsize)
        ax.set_ylabel(y_label, fontsize=fontsize)
        ax.tick_params(axis='both', which='major', labelsize=fontsize)
        if x_lim is None: # and len(x_lim) != 2
            pass
        else:
            ax.set_xlim(left=x_lim[0], right=x_lim[1])
        
        if y_lim is None: # and len(y_lim) != 2
            pass
        else:
            ax.set_ylim(bottom=y_lim[0], top=y_lim[1])

        plt.tight_layout()
        if show:
            plt.show()
        if save:
            plt.savefig(out_path)
        plt.close()
    return r, p

def hexplot(x, y, show=False, save=False,
            out_path=None, 
            fig_width=5, fig_height=5,
            x_label='', y_label='',
            fontsize=14):
    r = pearsonr(x, y)[0]
    p = pearsonr(x, y)[1]
    if show or save:
        _, ax = plt.subplots(figsize=(float(fig_width), float(fig_height)))
        g = sns.jointplot(x=x, y=y, kind="hex")
        sns.regplot(x=x,y=y, robust=True, scatter=False,
                            ax=g.ax_joint)
        g.ax_marg_x.set_title('r={:.2f}, p={:.2e}'.format(r, p), fontdict={'fontsize': fontsize})
        g.ax_joint.set_xlabel(x_label, fontsize=fontsize)
        g.ax_joint.set_ylabel(y_label, fontsize=fontsize)
        g.ax_joint.tick_params(axis='both', which='major', labelsize=fontsize)
        plt.tight_layout()
        if show:
            plt.show()
        if save:
            plt.savefig(out_path)
        plt.close()
    return r, p