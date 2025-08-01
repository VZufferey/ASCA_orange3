import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, RegressorMixin, BaseEstimator
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
import matplotlib.cm as cm
from scipy.spatial import ConvexHull
import math
import itertools


class ASCA(BaseEstimator):
    def __init__(self):
        self.__name__ = 'ASCA'

        # Default
        self.factors = None
        self.interactions = None
        self.data = None
        self.design = None
        self.effects = None
        self.factor_names = None #VZ
        self.variable_names = None  #VZ

        # Total (average)
        self.residuals = None
        self.total_factors = None
        self._total_interactions = None

    @staticmethod
    def svd_signstable(X):
        ###Based on Bro, R., Acar, E. and Kolda, T.G., 2008. Resolving the sign ambiguity in the singular value decomposition. Journal of Chemometrics: A Journal of the Chemometrics Society, 22(2), pp.135-140.
        try:
            X = np.asarray(X)
        except:
            pass
        U, D, V = np.linalg.svd(X, full_matrices=False)

        V = V.T  # python V is transposed compared to matlab
        K = len(D)
        s_left = np.zeros((1, K))

        # step 1
        for k in range(K):
            select = np.setdiff1d(list(range(K)), k)
            DD = np.zeros((K - 1, K - 1))
            np.fill_diagonal(DD, D[select])
            Y = X - U[:, select] @ DD @ V[:, select].T

            s_left_parts = np.zeros((1, Y.shape[1]))

            for j in range(Y.shape[1]):
                temp_prod = (U[:, k].T) @ (Y[:, j])
                s_left_parts[:, j] = (np.sign(temp_prod) + (temp_prod == 0)) * (temp_prod ** 2)

            s_left[:, k] = np.sum(s_left_parts)

        # step 2
        s_right = np.zeros((1, K))
        for k in range(K):
            select = np.setdiff1d(list(range(K)), k)
            DD = np.zeros((K - 1, K - 1))
            np.fill_diagonal(DD, D[select])
            Y = X - U[:, select] @ DD @ V[:, select].T

            s_right_parts = np.zeros((1, Y.shape[0]))
            for i in range(Y.shape[0]):
                temp_prod = (V[:, k].T) @ (Y[i, :].T)
                s_right_parts[:, i] = (np.sign(temp_prod) + (temp_prod == 0)) * (temp_prod ** 2)
            s_right[:, k] = np.sum(s_right_parts)

            # step 3
        for k in range(K):
            if (s_right[:, k] * s_left[:, k]) < 0:
                if s_left[:, k] < s_right[:, k]:
                    s_left[:, k] = -s_left[:, k]
                else:
                    s_right[:, k] = -s_right[:, k]
        left = np.zeros((K, K))
        right = np.zeros((K, K))
        np.fill_diagonal(left, np.sign(s_left) + (s_left == 0))
        np.fill_diagonal(right, np.sign(s_right) + (s_right == 0))
        U = U @ left
        V = V @ right
        return U, D, V

    @classmethod
    def do_PCA(self, _X, _Res):
        u, sv, v = self.svd_signstable(_X)
        # u,sv,v=np.linalg.svd(_X,full_matrices=False)
        scores = u * sv
        singular_values = np.zeros((len(sv), len(sv)))
        np.fill_diagonal(singular_values, sv)
        # singular_values=np.diag(sv)
        loadings = v
        projected = _Res @ v
        explained = (sv ** 2) / (np.sum(sv ** 2) * 100 + np.finfo(float).eps)
        return scores, loadings, projected, singular_values, explained

    def fit(self, X, y, interactions=[None], factor_names=None, variable_names=None): #VZ
        # initialize for SCA/PCA
        # scores
        self.factors_scores = []
        self.interaction_scores = []
        # loadings
        self.factors_loadings = []
        self.interaction_loadings = []
        # projected
        self.factors_projected = []
        self.interaction_projected = []
        # singular
        self.factors_singular = []
        self.interaction_singular = []
        # explained (variance)
        self.factors_explained = []
        self.interaction_explained = []

        # Save input data
        self.data = X
        self.design = y

        # Names
        self.factor_names = factor_names
        self.variable_names = variable_names

        # Mean center data
        Xmean = np.mean(X, axis=0)
        Xm = X - Xmean
        # Save as F (sklearn syntax issue)
        F = y

        # Make a set of unique factors
        factor_set = set(F.flatten())

        # Prepare a zero matrix corresponding to Xm
        zero = np.zeros_like(Xm)

        # Calculate Effects
        X_effect = []
        for effect in range(F.shape[1]):
            X_effect.append(np.zeros_like(Xm))
            for f in factor_set:
                select = F[:, effect] == f
                select_mat = np.where([select] * Xm.T.shape[0], Xm.T, zero.T).T
                select_avg = np.sum(select_mat, axis=0) / (
                            sum(select) + np.finfo(float).eps)  # average avoiding empty entry
                select_mat[select_mat.nonzero()] = 1  # set all as 1
                X_effect[effect] = X_effect[effect] + select_mat * select_avg

        # Sum the effects
        Total_effect = []
        for Xe in X_effect:
            if len(Total_effect) == 0:
                Total_effect = Xe
            else:
                Total_effect = Total_effect + Xe

                # Calculate Interactions
        X_interact = []

        for inter, idxs in enumerate(interactions):
            X_inter = np.zeros_like(Xm)
            idxs = list(idxs)
            # Generates combinnation for THIS interaction
            group_labels = [tuple(row[idxs]) for row in F]
            unique_labels = list(set(group_labels))
            for ulabel in unique_labels:
                mask = np.array([tuple(row[idxs]) == ulabel for row in F])
                if not np.any(mask):
                    continue
                # Average observation in the  interaction group
                group_mean = Xm[mask].mean(axis=0)
                # Sum of main effects
                sum_main = np.zeros_like(group_mean)
                for j, idx in enumerate(idxs):
                    fac_val = ulabel[j]
                    mask_fac = (F[:, idx] == fac_val)
                    sum_main += X_effect[idx][mask_fac].mean(axis=0)
                X_inter[mask] = group_mean - sum_main
            X_interact.append(X_inter)
            ########

        # Sum the Interactions
        Total_interact = []
        for Xi in X_interact:
            if len(Total_interact) == 0:
                Total_interact = Xi
            else:
                Total_interact = Total_interact + Xi

        # Calculate Residual/Error
        E = Xm - np.mean(Xm, axis=0) - Total_effect - Total_interact

        # Percentage Effects calculation
        SSQ_X = np.sum(X * X)
        SSQ_mean = np.sum(np.tile(Xmean, (X.shape[0], 1)) ** 2)

        SSQ_factors = [np.sum(Xe ** 2) for Xe in X_effect]  # for each individual factors

        SSQ_residuals = np.sum(E ** 2)
        SSQ_interactions = np.sum(Total_interact ** 2)
        SSQ = np.asarray([SSQ_mean, *SSQ_factors, SSQ_interactions, SSQ_residuals])
        percentage_effect = SSQ / SSQ_X * 100

        # Save all information
        self.factors = X_effect
        self.interactions = X_interact
        self.total_factors = Total_effect
        self._total_interactions = Total_interact
        self.effects = percentage_effect
        self.residuals = E

        # SCA
        for Xe in X_effect:
            _scores, _loadings, _projected, _singular_values, _explained = self.do_PCA(Xe, E)
            self.factors_scores.append(_scores)
            self.factors_loadings.append(_loadings)
            self.factors_projected.append(_projected)
            self.factors_singular.append(_singular_values)
            self.factors_explained.append(_explained)
        #Store these values as summary for effect subtract

        for Xi in X_interact:
            _scores, _loadings, _projected, _singular_values, _explained = self.do_PCA(Xi, E)
            self.interaction_scores.append(_scores)
            self.interaction_loadings.append(_loadings)
            self.interaction_projected.append(_projected)
            self.interaction_singular.append(_singular_values)
            self.interaction_explained.append(_explained)
        self.factors_scores[1][:, 1] = -self.factors_scores[1][:, 1]

    # enhanced interactions plotting methodd (stacked bars instead of interleaved)
    def plot_interactions(self, interactions=None): #, factor_names=None, variable_names=None
        #Factors interactions
        for ii in range(len(self.interaction_scores)):
            fig, ax = plt.subplots(1, 2)
            inter_idx = interactions[ii] if interactions and ii < len(interactions) else [ii]
            inter_names = [self.factor_names[i] for i in inter_idx] if self.factor_names else [f"Interaction {ii + 1}"]
            combis = [tuple(row[inter_idx]) for row in self.design]
            unique_combi = list(dict.fromkeys(combis))
            cmap = plt.get_cmap('tab20')
            colors = [cmap(i % 20) for i in range(len(unique_combi))]
            for i, combi in enumerate(unique_combi):
                indices = [j for j, tup in enumerate(combis) if tup == combi]
                if len(indices) == 0:
                    continue
                group_scores = self.interaction_scores[ii][indices]
                bary = group_scores.mean(axis=0)
                # Scatter du barycentre
                ax[0].scatter(bary[0], bary[1], c=[colors[i]], marker=MarkerStyle('o', fillstyle='none'), s=80,
                              label=" / ".join(map(str, combi)))
                # Bounding box if enough points
                if group_scores.shape[0] >= 3:
                    try:
                        hull = ConvexHull(group_scores[:, :2])  # On prend PC1 & PC2
                        hull_points = group_scores[hull.vertices]
                        ax[0].plot(
                            np.append(hull_points[:, 0], hull_points[0, 0]),
                            np.append(hull_points[:, 1], hull_points[0, 1]),
                            color=colors[i], linestyle='--', linewidth=0.7
                        )
                    except Exception:
                        pass  #  In case ConveHull fails
            inter_title = " × ".join(inter_names)
            ax[0].set_xlabel(f'PC 1 ({round(self.interaction_explained[ii][0], 2)}% expl. variance)')
            ax[0].set_ylabel(f'PC 2 ({round(self.interaction_explained[ii][1], 2)}% expl. variance)')
            ax[0].set_title(f'Interaction : {inter_title}')
            ax[0].legend(title=inter_title, loc='best', fontsize='small')

            # Loadings
            first_loading = self.interaction_loadings[ii][:, 0]
            second_loading = self.interaction_loadings[ii][:, 1]
            if self.variable_names and len(self.variable_names) == len(first_loading):
                labels = self.variable_names
            else:
                labels = [f"Var {_i + 1}" for _i in range(len(first_loading))]
            x = np.arange(len(labels))
            width = 0.6  # plus large pour donner de la lisibilité

            # Découpe en positifs et négatifs
            first_pos = np.clip(first_loading, 0, None)
            first_neg = np.clip(first_loading, None, 0)
            second_pos = np.clip(second_loading, 0, None)
            second_neg = np.clip(second_loading, None, 0)

            # Barres empilées : PC1, puis PC2 "stacked" dessus pour chaque variable
            ax[1].bar(x, first_neg, width, label='PC1 (neg)', color='tab:blue')
            ax[1].bar(x, first_pos, width, label='PC1 (pos)', color='tab:blue')
            ax[1].bar(x, second_neg, width, bottom=first_neg, label='PC2 (neg)', color='tab:orange')
            ax[1].bar(x, second_pos, width, bottom=first_pos, label='PC2 (pos)', color='tab:orange')

            ax[1].hlines(y=0, xmin=-width - 0.05, xmax=len(labels) - 1 + width + 0.01 * 5, linewidth=0.5,
                         linestyles='--', color='black')
            ax[1].set_ylabel('Value')
            ax[1].set_title(f'Loadings ({inter_title})')
            ax[1].set_xticks(x)
            ax[1].set_xticklabels(labels, rotation=90, fontsize=8, ha='center')
            ax[1].legend()
            plt.tight_layout()
            plt.show()

    #Vanilla Interactions plotting fucntion
    def plot_interactions1(self, factor_names=None, variable_names=None):
        for ii in range(len(self.interaction_scores)):
            # scores
            fig, ax = plt.subplots(1, 2)
            ax[0].scatter(self.interaction_scores[ii][:, 0], self.interaction_scores[ii][:, 1], color='r')
            ax[0].scatter(self.interaction_projected[ii][:, 0], self.interaction_projected[ii][:, 1], c='blue',
                          marker=MarkerStyle('o', fillstyle='none'))
            n = [" ".join(list(map(str, x))) for x in self.design]

            #for i, txt in enumerate(n):
               # ax[0].annotate(txt, (self.interaction_scores[ii][i, 0], self.interaction_scores[ii][i, 1]))
            ax[0].set_xlabel('PC 1 (' + str(round(self.interaction_explained[ii][0], 2)) + '% expl. variance)')
            ax[0].set_ylabel('PC 2 (' + str(round(self.interaction_explained[ii][1], 2)) + '% expl. variance)')
            ax[0].set_title('Interaction=' + str(ii + 1))

            # loadings
            first_loading = self.interaction_loadings[ii][:, 0]
            second_loading = self.interaction_loadings[ii][:, 1]
            if variable_names is not None and len(variable_names) == self.factors_loadings[ii].shape[0]:
                labels = list(variable_names)
            else:  # Generic name (Vanilla)
                labels = [f"Response {_i + 1}" for _i in range(self.factors_loadings[ii].shape[0])]
            x = np.arange(len(labels))
            width = 0.2
            my_cmap = list(plt.get_cmap("Set1").colors)
            rects1 = ax[1].bar(x - width / 2 - 0.01, first_loading, width, label='First Loading', color=my_cmap[0])
            rects2 = ax[1].bar(x + width / 2 + 0.01, second_loading, width, label='Second Loading', color=my_cmap[1])
            ax[1].hlines(y=0, xmin=0 - width - 0.05, xmax=len(labels) - 1 + width + 0.01 * 5, linewidth=0.5,
                         linestyles='--', color='black')
            ax[1].set_ylabel('Value')
            ax[1].set_title('Loading for Factor ' + str(ii + 1))
            ax[1].set_xticks(x, labels)
            ax[1].set_xticklabels(labels, rotation=90, fontsize=8, ha='right')
            ax[1].legend()
            plt.tight_layout()
            plt.show()

    def plot_factors(self): #factor_names=[None], variable_names=[None]
        print("Factor names in plot_factors() : ", self.factor_names)
        for ii in range(len(self.factors_scores)):
            yy = self.design[:, ii]
            set_yy = list(dict.fromkeys(yy))  # VZ keep order of apparition

            #Setting factor names for titles
            if self.factor_names is not None and ii < len(self.factor_names):
                factor_name = self.factor_names[ii]
            else:
                factor_name = f"Factor {ii + 1}"
            print("Plotting Factor : ", factor_name)

            # fig, ax = plt.subplots(1, 2, constrained_layout=True)
            nb_groups = len(set_yy)
            cmap = plt.get_cmap('tab20')  # VZ
            colors = [cmap(i % 20) for i in range(nb_groups)]  # VZ larger color map, wrapped.
            fig, ax = plt.subplots(
                1, 2,
                figsize=(12, 6),
                gridspec_kw={'width_ratios': [1, 1]},
                constrained_layout=True
            )
            fig.canvas.manager.set_window_title(f"ASCA_plot_{factor_name}")

            ########################################
            #Factors
            for i, _yy in enumerate(set_yy):
                plot_x = []
                plot_y = []
                plot_colors = []
                plot_label = []

                plot_sx = []
                plot_sy = []

                ###VZ
                # index values equal to _yy
                mask = (yy == _yy)
                plot_x = (self.factors_scores[ii][mask, 0] + self.factors_projected[ii][mask, 0]).tolist()
                plot_y = (self.factors_scores[ii][mask, 1] + self.factors_projected[ii][mask, 1]).tolist()
                if plot_x:  # S'il y a des points à afficher
                    ax[0].scatter(
                        plot_x, plot_y,
                        c=[colors[i]] * len(plot_x),
                        marker=MarkerStyle('o', fillstyle='none'),
                        label=str(_yy)  # Affiche la valeur réelle comme étiquette
                )### VZ end

                # Boundary box
                points = np.stack((plot_x, plot_y), axis=1)
                if points.shape[0] >= 3:
                    hull = ConvexHull(points)
                    for simplex in hull.simplices:
                        ax[0].plot(points[simplex, 0], points[simplex, 1],
                                   color=colors[i], linestyle='--', linewidth=0.3)

            n = [" ".join(list(map(str, x))) for x in self.design]

            '''
            for i, txt in enumerate(n):
                ax[0].annotate(txt, (self.factors_scores[ii][i,0], self.factors_scores[ii][i,1]))
            '''
            ########################################################
            #PCA scatter plot
            ax[0].set_xlabel('PC 1 (' + str(round(self.factors_explained[ii][0], 2)) + '% expl. variance)')
            ax[0].set_ylabel('PC 2 (' + str(round(self.factors_explained[ii][1], 2)) + '% expl. variance)')
            #ax[0].set_title('Factor ' + str(ii + 1))
            ax[0].set_title(str(factor_name)) #VZ
            handles, labels = ax[0].get_legend_handles_labels()
            nb_col = min(5, len(labels))
            ax[0].legend(#title=self.factor_names[ii],
                         loc='upper center',
                         ncol = nb_col,
                         bbox_to_anchor=(0.5, -0.15),
                         fontsize='small', title_fontsize='medium',
                         frameon=False, borderaxespad=0
                         ) #VZ

            ########################################################
            # loadings bar plots
            first_loading = self.factors_loadings[ii][:, 0]
            second_loading = self.factors_loadings[ii][:, 1]
            if self.variable_names is not None and len(self.variable_names) == self.factors_loadings[ii].shape[0]:
                labels = list(self.variable_names)
            else: # Generic name (Vanilla)
                labels = ["Response {_i + 1}" for _i in range(self.factors_loadings[ii].shape[0])]

            x = np.arange(len(labels))
            width = 0.2
            my_cmap = list(plt.get_cmap("Set1").colors)
            #rects1 = ax[1].bar(x - width / 2 - 0.01, first_loading, width, label='First Loading (PC1)', color=my_cmap[0])
            #rects2 = ax[1].bar(x + width / 2 + 0.01, second_loading, width, label='Second Loading (PC2)', color=my_cmap[1])
            first_pos = np.clip(first_loading, 0, None)
            first_neg = np.clip(first_loading, None, 0)
            second_pos = np.clip(second_loading, 0, None)
            second_neg = np.clip(second_loading, None, 0)
            # Barres négatives et positives empilées pour PC1 et PC2
            ax[1].bar(x, first_neg, width, label='PC1 (neg)', color='tab:blue')
            ax[1].bar(x, first_pos, width, label='PC1 (pos)', color='tab:blue')
            ax[1].bar(x, second_neg, width, bottom=first_neg, label='PC2 (neg)', color='tab:orange')
            ax[1].bar(x, second_pos, width, bottom=first_pos, label='PC2 (pos)', color='tab:orange')

            ax[1].hlines(y=0, xmin=0 - width - 0.05, xmax=len(labels) - 1 + width + 0.01 * 5, linewidth=0.5,
                         linestyles='--', color='black')
            ax[1].set_ylabel('Value')
            #ax[1].set_title('Loading for Factor ' + str(ii + 1))
            ax[1].set_title(f'Loading for {factor_name}') #VZ
            ax[1].set_xticks(x, labels)
            ax[1].set_xticklabels(labels, rotation=90, fontsize=8, ha='right')
            ax[1].legend()
            #plt.tight_layout()
            #plt.subplots_adjust(bottom=0.25)  # ou adapte à 0.3, 0.35 selon
            plt.show()

    def plot_factors_biplot(self):
        labels = None
        sign = lambda x: x and 1 - 2 * (x < 0)
        for ii in range(len(self.factors_scores)):
            xs = self.factors_scores[ii][:, 0]
            ys = self.factors_scores[ii][:, 1]
            coeff = self.factors_loadings[ii]
            n = len(coeff)
            scalex = 1 / (xs.max() - xs.min())
            scaley = 1 / (ys.max() - ys.min())
            plt.scatter(xs * scalex, ys * scaley, c=self.design[:, ii], cmap='Set1')
            for i in range(n):
                plt.arrow(0, 0, coeff[i, 0], coeff[i, 1], color='black', alpha=0.3, head_starts_at_zero=False,
                          head_width=0.03)
                if labels is None:
                    plt.text(coeff[i, 0] + sign(coeff[i, 0]) * 0.05, coeff[i, 1] + sign(coeff[i, 1]) * 0.05,
                             "Var" + str(i + 1), color='darkgreen', ha='center', va='center')
                else:
                    plt.text(coeff[i, 0] + sign(coeff[i, 0]) * 0.05, coeff[i, 1] + sign(coeff[i, 1]) * 0.05, labels[i],
                             color='darkgreen', ha='center', va='center')
            # plt.xlim(-1,1)
            # plt.ylim(-1,1)
            plt.grid(color='grey', linestyle='--', linewidth=0.5)
            plt.xlabel("PC1 (" + str(round(self.factors_explained[ii][0], 2)) + "%)")
            plt.ylabel("PC2 (" + str(round(self.factors_explained[ii][1], 2)) + "%)")
            plt.show()

        # Make a table for the Scatter PC plots of a given factor

    # Function to save summary of the fit function (residuals, scores, etc)
    def getFitSummary(self):
        return (self.factors_scores,
                self.factors_loadings,
                self.factors_projected,
                self.factors_singular,
                self.factors_explained)

    # Function to save summary of the fit function (residuals, scores, etc)
    def getEffects(self, factor=None):
        print("factor: ",factor,", (",self.factor_names[factor],")")
        if factor is not None:
            return self.factors[factor]
        else:
            return self.factors

    # Function to save the PCA scatter plots data (xfor further replotting in other plotting softwares)
    def getData_PCplot(self, Factor=None):
        if Factor is None or Factor >= len(self.factors_scores):
            raise ValueError('Factor argument is None or out of bounds')

        # Gathering scores and projections
        scores = self.factors_scores[Factor]
        projected = self.factors_projected[Factor]

        # Group values
        groups = self.design[:, Factor]

        # C1/PC2 coordinates calculati9on
        pc1 = scores[:, 0] + projected[:, 0]
        pc2 = scores[:, 1] + projected[:, 1]

        # Making final table.
        group_col_name = self.factor_names[Factor] if self.factor_names and len(
            self.factor_names) > Factor else f"Group_{Factor}"
        data_PCplot = pd.DataFrame({
            'PC1': pc1,
            'PC2': pc2,
            group_col_name: groups
        })
        return data_PCplot

if __name__ == '__main__':
    X = [[1.0000, 0.6000],
         [3.0000, 0.4000],
         [2.0000, 0.7000],
         [1.0000, 0.8000],
         [2.0000, 0.0100],
         [2.0000, 0.8000],
         [4.0000, 1.0000],
         [6.0000, 2.0000],
         [5.0000, 0.9000],
         [5.0000, 1.0000],
         [6.0000, 2.0000],
         [5.0000, 0.7000]]
    X = np.asarray(X)

    F = [[1, 1],
         [1, 1],
         [1, 2],
         [1, 2],
         [1, 3],
         [1, 3],
         [2, 1],
         [2, 1],
         [2, 2],
         [2, 2],
         [2, 3],
         [2, 3]]
    F = np.asarray(F)
    interactions = [[0, 1]]

    ASCA = ASCA()
    ASCA.fit(X, F, interactions, [["Factor A", "Factor B"]])
    ASCA.plot_factors()
    ASCA.plot_interactions()
    ASCA.plot_factors_biplot()
    # print(ASCA.factors_scores)
    # print(ASCA.factors_loadings)
    # print(ASCA.factors_projected)
    # ang=math.radians(90)
    # A=np.asarray([[math.cos(ang),-math.sin(ang)],[math.sin(ang),math.cos(ang)]])
    # print([ f@A for f  in ASCA.factors_loadings])
