import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import spearmanr
import articleHandler as handler

#  SELECT CROWDWORKER GROUP
crowdworker_group = "default"
# crowdworker_group = "sowj"  # not enough judgements, less tan 10 crowdworkers!
# crowdworker_group = "west"
# crowdworker_group = "rest"

articles, countries = handler.get_articles(crowdworker_group)

# not all judgements are included, it is not possible to calculate article correlations
ONLY_SENTENCES = crowdworker_group != "default"

# mode='maj'
# mode='intensified'
mode = 'avg'

if not ONLY_SENTENCES:
    article_dimensions_dict = {
        "Subj.": [a.scores['subj'][mode] for a in articles],
        "Hidden Assumpt.": [a.scores['hidden-assumpt'][mode] for a in articles],
        "Russia Neg.":  [a.scores['framing']['russia'][mode][2][1] for a in articles],
        "Russia Neutral":  [a.scores['framing']['russia'][mode][1][1] for a in articles],
        "Russia Pos.":  [a.scores['framing']['russia'][mode][0][1] for a in articles],
        "Ukraine Neg.":  [a.scores['framing']['ukraine'][mode][2][1] for a in articles],
        "Ukraine Neutral":  [a.scores['framing']['ukraine'][mode][1][1] for a in articles],
        "Ukraine Pos.":  [a.scores['framing']['ukraine'][mode][0][1] for a in articles],
        "West Neg.":  [a.scores['framing']['west'][mode][2][1] for a in articles],
        "West Neutral":  [a.scores['framing']['west'][mode][1][1] for a in articles],
        "West Pos.":  [a.scores['framing']['west'][mode][0][1] for a in articles],
        # "West+Ukr Neg.":  [(a.scores['framing']['west'][mode][2][1] + a.scores['framing']['ukraine'][mode][2][1]) / 2 for a in articles],
        # "West+Ukr Neutral":  [(a.scores['framing']['west'][mode][1][1] + a.scores['framing']['ukraine'][mode][1][1]) / 2 for a in articles],
        # "West+Ukr Pos.":  [(a.scores['framing']['west'][mode][0][1] + a.scores['framing']['ukraine'][mode][0][1]) / 2 for a in articles],
        "Bias Rus.": [a.scores['bias']['pro-russia'][mode] for a in articles],
        "Bias West": [a.scores['bias']['pro-west'][mode] for a in articles],
        "Biased in General\n(Rus. or West)": [int((a.scores['bias']['pro-west'][mode] != 0) or (a.scores['bias']['pro-russia'][mode] != 0)) for a in articles],
    }

    article_inner_dataframe = pd.DataFrame(article_dimensions_dict)

    article_proRussia_dataframe = pd.DataFrame({
        "Subj.": [a.scores['subj'][mode] for a in articles if a.leaning in [0, -1]],
        "Hidden Assumpt.": [a.scores['hidden-assumpt'][mode] for a in articles if a.leaning in [0, -1]],
        "Russia Neg.":  [a.scores['framing']['russia'][mode][2][1] for a in articles if a.leaning in [0, -1]],
        "Russia Neutral":  [a.scores['framing']['russia'][mode][1][1] for a in articles if a.leaning in [0, -1]],
        "Russia Pos.":  [a.scores['framing']['russia'][mode][0][1] for a in articles if a.leaning in [0, -1]],
        "Ukraine Neg.":  [a.scores['framing']['ukraine'][mode][2][1] for a in articles if a.leaning in [0, -1]],
        "Ukraine Neutral":  [a.scores['framing']['ukraine'][mode][1][1] for a in articles if a.leaning in [0, -1]],
        "Ukraine Pos.":  [a.scores['framing']['ukraine'][mode][0][1] for a in articles if a.leaning in [0, -1]],
        "West Neg.":  [a.scores['framing']['west'][mode][2][1] for a in articles if a.leaning in [0, -1]],
        "West Neutral":  [a.scores['framing']['west'][mode][1][1] for a in articles if a.leaning in [0, -1]],
        "West Pos.":  [a.scores['framing']['west'][mode][0][1] for a in articles if a.leaning in [0, -1]],
        "Bias Rus.": [a.scores['bias']['pro-russia'][mode] for a in articles if a.leaning in [0, -1]],
        "Bias West": [a.scores['bias']['pro-west'][mode] for a in articles if a.leaning in [0, -1]],
        "Biased in General\n(Rus. or West)": [int((a.scores['bias']['pro-west'][mode] != 0) or (a.scores['bias']['pro-russia'][mode] != 0)) for a in articles if a.leaning in [0, -1]],
        "Pro-Russia (Expert)":  [int(a.leaning != 0) for a in articles if a.leaning in [0, -1]]
    })

    article_proWest_dataframe = pd.DataFrame({
        "Subj.": [a.scores['subj'][mode] for a in articles if a.leaning in [0, 1]],
        "Hidden Assumpt.": [a.scores['hidden-assumpt'][mode] for a in articles if a.leaning in [0, 1]],
        "Russia Neg.":  [a.scores['framing']['russia'][mode][2][1] for a in articles if a.leaning in [0, 1]],
        "Russia Neutral":  [a.scores['framing']['russia'][mode][1][1] for a in articles if a.leaning in [0, 1]],
        "Russia Pos.":  [a.scores['framing']['russia'][mode][0][1] for a in articles if a.leaning in [0, 1]],
        "Ukraine Neg.":  [a.scores['framing']['ukraine'][mode][2][1] for a in articles if a.leaning in [0, 1]],
        "Ukraine Neutral":  [a.scores['framing']['ukraine'][mode][1][1] for a in articles if a.leaning in [0, 1]],
        "Ukraine Pos.":  [a.scores['framing']['ukraine'][mode][0][1] for a in articles if a.leaning in [0, 1]],
        "West Neg.":  [a.scores['framing']['west'][mode][2][1] for a in articles if a.leaning in [0, 1]],
        "West Neutral":  [a.scores['framing']['west'][mode][1][1] for a in articles if a.leaning in [0, 1]],
        "West Pos.":  [a.scores['framing']['west'][mode][0][1] for a in articles if a.leaning in [0, 1]],
        "Bias Rus.": [a.scores['bias']['pro-russia'][mode] for a in articles if a.leaning in [0, 1]],
        "Bias West": [a.scores['bias']['pro-west'][mode] for a in articles if a.leaning in [0, 1]],
        "Biased in General\n(Rus. or West)": [int((a.scores['bias']['pro-west'][mode] != 0) or (a.scores['bias']['pro-russia'][mode] != 0)) for a in articles if a.leaning in [0, 1]],
        "Pro-West (Expert)":  [int(a.leaning != 0) for a in articles if a.leaning in [0, 1]]
    })

    article_binary_leaning_dataframe = pd.DataFrame({
        "Subj.": [a.scores['subj'][mode] for a in articles],
        "Hidden Assumpt.": [a.scores['hidden-assumpt'][mode] for a in articles],
        "Russia Neg.":  [a.scores['framing']['russia'][mode][2][1] for a in articles],
        "Russia Neutral":  [a.scores['framing']['russia'][mode][1][1] for a in articles],
        "Russia Pos.":  [a.scores['framing']['russia'][mode][0][1] for a in articles],
        "Ukraine Neg.":  [a.scores['framing']['ukraine'][mode][2][1] for a in articles],
        "Ukraine Neutral":  [a.scores['framing']['ukraine'][mode][1][1] for a in articles],
        "Ukraine Pos.":  [a.scores['framing']['ukraine'][mode][0][1] for a in articles],
        "West Neg.":  [a.scores['framing']['west'][mode][2][1] for a in articles],
        "West Neutral":  [a.scores['framing']['west'][mode][1][1] for a in articles],
        "West Pos.":  [a.scores['framing']['west'][mode][0][1] for a in articles],
        "Bias Rus.": [a.scores['bias']['pro-russia'][mode] for a in articles],
        "Bias West": [a.scores['bias']['pro-west'][mode] for a in articles],
        "Biased in General\n(Rus. or West)": [int((a.scores['bias']['pro-west'][mode] != 0) or (a.scores['bias']['pro-russia'][mode] != 0)) for a in articles],
        "Biased (Expert)\n(Pro-Russia and Pro-West)":  [int(a.leaning != 0) for a in articles]
    })


def split_framing_scale_helper(scale, val):
    # this function simplifies the creation of the sentence dimensions dictionary (sent_dimensions_dict)

    if scale == 'neg':
        return abs(min(0, val))
    else:
        return max(0, val)


sent_inner_dataframe = pd.DataFrame({
    "Subj.": [sent.scores['subj'][mode] for a in articles for sent in a.sentences],
    "Hidden Assumpt.": [sent.scores['hidden-assumpt'][mode] for a in articles for sent in a.sentences],
    "Russia Neg.":  [split_framing_scale_helper('neg', sent.scores['framing']['russia'][mode]) for a in articles for sent in a.sentences],
    "Russia Pos.":  [split_framing_scale_helper('pos', sent.scores['framing']['russia'][mode]) for a in articles for sent in a.sentences],
    "Ukraine Neg.":  [split_framing_scale_helper('neg', sent.scores['framing']['ukraine'][mode]) for a in articles for sent in a.sentences],
    "Ukraine Pos.":  [split_framing_scale_helper('pos', sent.scores['framing']['ukraine'][mode]) for a in articles for sent in a.sentences],
    "West Neg.":  [split_framing_scale_helper('neg', sent.scores['framing']['west'][mode]) for a in articles for sent in a.sentences],
    "West Pos.":  [split_framing_scale_helper('pos', sent.scores['framing']['west'][mode]) for a in articles for sent in a.sentences],
    # "West+Ukr Neg.":  [(split_framing_scale_helper('neg', sent.scores['framing']['west'][mode]) + split_framing_scale_helper('neg', sent.scores['framing']['ukraine'][mode])) / 2 for a in articles for sent in a.sentences],
    # "West+Ukr Pos.":  [(split_framing_scale_helper('pos', sent.scores['framing']['west'][mode]) + split_framing_scale_helper('pos', sent.scores['framing']['west'][mode])) / 2 for a in articles for sent in a.sentences],
    "Bias Rus.": [sent.scores['bias']['pro-russia'][mode] for a in articles for sent in a.sentences],
    "Bias West": [a.scores['bias']['pro-west'][mode] for a in articles for sent in a.sentences],
    "Biased in General\n(Rus. or West)": [int((a.scores['bias']['pro-west'][mode] != 0) or (a.scores['bias']['pro-russia'][mode] != 0)) for a in articles for sent in a.sentences],
})

sent_proRussia_dataframe = pd.DataFrame({
    "Subj.": [sent.scores['subj'][mode] for a in articles for sent in a.sentences if a.leaning in [0, -1]],
    "Hidden Assumpt.": [sent.scores['hidden-assumpt'][mode] for a in articles for sent in a.sentences if a.leaning in [0, -1]],
    "Russia Neg.":  [split_framing_scale_helper('neg', sent.scores['framing']['russia'][mode]) for a in articles for sent in a.sentences if a.leaning in [0, -1]],
    "Russia Pos.":  [split_framing_scale_helper('pos', sent.scores['framing']['russia'][mode]) for a in articles for sent in a.sentences if a.leaning in [0, -1]],
    "Ukraine Neg.":  [split_framing_scale_helper('neg', sent.scores['framing']['ukraine'][mode]) for a in articles for sent in a.sentences if a.leaning in [0, -1]],
    "Ukraine Pos.":  [split_framing_scale_helper('pos', sent.scores['framing']['ukraine'][mode]) for a in articles for sent in a.sentences if a.leaning in [0, -1]],
    "West Neg.":  [split_framing_scale_helper('neg', sent.scores['framing']['west'][mode]) for a in articles for sent in a.sentences if a.leaning in [0, -1]],
    "West Pos.":  [split_framing_scale_helper('pos', sent.scores['framing']['west'][mode]) for a in articles for sent in a.sentences if a.leaning in [0, -1]],
    "Bias Rus.": [sent.scores['bias']['pro-russia'][mode] for a in articles for sent in a.sentences if a.leaning in [0, -1]],
    "Bias West": [a.scores['bias']['pro-west'][mode] for a in articles for sent in a.sentences if a.leaning in [0, -1]],
    "Biased in General\n(Rus. or West)": [int((a.scores['bias']['pro-west'][mode] != 0) or (a.scores['bias']['pro-russia'][mode] != 0)) for a in articles for sent in a.sentences if a.leaning in [0, -1]],
    "Pro-Russia (Expert)":  [int(a.leaning != 0) for a in articles for sent in a.sentences if a.leaning in [0, -1]]
})

sent_proWest_dataframe = pd.DataFrame({
    "Subj.": [sent.scores['subj'][mode] for a in articles for sent in a.sentences if a.leaning in [0, 1]],
    "Hidden Assumpt.": [sent.scores['hidden-assumpt'][mode] for a in articles for sent in a.sentences if a.leaning in [0, 1]],
    "Russia Neg.":  [split_framing_scale_helper('neg', sent.scores['framing']['russia'][mode]) for a in articles for sent in a.sentences if a.leaning in [0, 1]],
    "Russia Pos.":  [split_framing_scale_helper('pos', sent.scores['framing']['russia'][mode]) for a in articles for sent in a.sentences if a.leaning in [0, 1]],
    "Ukraine Neg.":  [split_framing_scale_helper('neg', sent.scores['framing']['ukraine'][mode]) for a in articles for sent in a.sentences if a.leaning in [0, 1]],
    "Ukraine Pos.":  [split_framing_scale_helper('pos', sent.scores['framing']['ukraine'][mode]) for a in articles for sent in a.sentences if a.leaning in [0, 1]],
    "West Neg.":  [split_framing_scale_helper('neg', sent.scores['framing']['west'][mode]) for a in articles for sent in a.sentences if a.leaning in [0, 1]],
    "West Pos.":  [split_framing_scale_helper('pos', sent.scores['framing']['west'][mode]) for a in articles for sent in a.sentences if a.leaning in [0, 1]],
    "Bias Rus.": [sent.scores['bias']['pro-russia'][mode] for a in articles for sent in a.sentences if a.leaning in [0, 1]],
    "Bias West.": [a.scores['bias']['pro-west'][mode] for a in articles for sent in a.sentences if a.leaning in [0, 1]],
    "Biased in General\n(Rus. or West)": [int((a.scores['bias']['pro-west'][mode] != 0) or (a.scores['bias']['pro-russia'][mode] != 0)) for a in articles for sent in a.sentences if a.leaning in [0, 1]],
    "Pro-West (Expert)":  [int(a.leaning != 0) for a in articles for sent in a.sentences if a.leaning in [0, 1]]
})

sent_binary_leaning_dataframe = pd.DataFrame({
    "Subj.": [sent.scores['subj'][mode] for a in articles for sent in a.sentences],
    "Hidden Assumpt.": [sent.scores['hidden-assumpt'][mode] for a in articles for sent in a.sentences],
    "Russia Neg.":  [split_framing_scale_helper('neg', sent.scores['framing']['russia'][mode]) for a in articles for sent in a.sentences],
    "Russia Pos.":  [split_framing_scale_helper('pos', sent.scores['framing']['russia'][mode]) for a in articles for sent in a.sentences],
    "Ukraine Neg.":  [split_framing_scale_helper('neg', sent.scores['framing']['ukraine'][mode]) for a in articles for sent in a.sentences],
    "Ukraine Pos.":  [split_framing_scale_helper('pos', sent.scores['framing']['ukraine'][mode]) for a in articles for sent in a.sentences],
    "West Neg.":  [split_framing_scale_helper('neg', sent.scores['framing']['west'][mode]) for a in articles for sent in a.sentences],
    "West Pos.":  [split_framing_scale_helper('pos', sent.scores['framing']['west'][mode]) for a in articles for sent in a.sentences],
    "Bias Rus.": [sent.scores['bias']['pro-russia'][mode] for a in articles for sent in a.sentences],
    "Bias West": [a.scores['bias']['pro-west'][mode] for a in articles for sent in a.sentences],
    "Biased in General\n(Rus. or West)": [int((a.scores['bias']['pro-west'][mode] != 0) or (a.scores['bias']['pro-russia'][mode] != 0)) for a in articles for sent in a.sentences],
    "Biased (Expert)\n(Pro-Russia and Pro-West)":  [int(a.leaning != 0) for a in articles for sent in a.sentences]
})


def get_inner_dimensional_dfs():
    # helper method to exclude article dataframes if necessary

    df = {
        "Sentence Bias Dimensions": sent_inner_dataframe
    }

    if not ONLY_SENTENCES:
        df["Article Bias Dimensions"] = article_inner_dataframe

    return df


def get_leaning_dfs():
    # helper method to exclude article dataframes if necessary

    df = {
        # Empty DataFrame
        "Sentence Bias Dimensions and pro-Russian Leaning (Expert)": sent_proRussia_dataframe,
        # Empty DataFrame
        "Sentence Bias Dimensions and pro-West Leaning (Expert)": sent_proWest_dataframe,
        # set(leaning) = {1} --> NaN
        "Sentence Bias Dimensions and Binary Leaning (Expert)": sent_binary_leaning_dataframe,
    }

    if not ONLY_SENTENCES:
        df["Article Bias Dimensions and pro-Russian Leaning (Expert)"] = article_proRussia_dataframe
        # Empty DataFrame
        df["Article Bias Dimensions and pro-West Leaning (Expert)"] = article_proWest_dataframe
        # set(leaning) = {1} --> NaN
        df["Article Bias Dimensions and Binary Leaning (Expert)"] = article_binary_leaning_dataframe

    return df


inner_dimensional_dfs = get_inner_dimensional_dfs()
leaning_dfs = get_leaning_dfs()


def create_colormap(name):
    # Be sure to select a color map excluding white!
    # ...Otherwise, no diagonal matrices can be plotted.

    my_cmap = cm.get_cmap(name, 10)
    my_cmap.set_bad('w')
    return my_cmap


def show_correlations(corr, title, large, diagonal, position, is_p):
    # this function plots all correlations

    if large:
        if diagonal:
            fig, ax = plt.subplots(figsize=(15, 15))
        else:
            fig, ax = plt.subplots(figsize=(18, 10))

    else:
        fig, ax = plt.subplots(figsize=(9, 9))

    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    my_cmap = (create_colormap('twilight_shifted'),
               create_colormap('magma'))[is_p]

    ticks = [-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1]

    divider = make_axes_locatable(ax)

    y_ticklabels = {"object": "", "rotation": ""}

    if diagonal:
        # mask out the upper triangle
        mask = np.tri(corr.shape[0], k=-1).transpose()
        diagonal_corr = np.ma.array(corr, mask=mask, dtype=float)
        im = ax.imshow(diagonal_corr, cmap=my_cmap, vmin=-1, vmax=1)

        y_ticklabels["object"] = corr.columns
        y_ticklabels["rotation"] = 45

    else:
        im = ax.imshow(corr, cmap=my_cmap, vmin=-1, vmax=1)

        y_ticklabels["object"] = corr.index
        y_ticklabels["rotation"] = 45

    ax.set_yticks(range(corr.shape[0]))
    ax.set_yticklabels(y_ticklabels["object"], rotation=y_ticklabels["rotation"],
                       ha='right', fontsize=15)

    ax.set_xticks(range(corr.shape[1]))
    ax.set_xticklabels(corr.columns, rotation=45, ha='right', fontsize=15)
    ax.tick_params(top=False, labelbottom=True, labeltop=False)

    if position == 'top':
        cax = divider.append_axes('top', size='8%', pad=0.3)
        cax.xaxis.set_ticks_position('top')
        cax.xaxis.set_label_position('top')
        fig.colorbar(im, cax=cax, ticks=ticks, orientation="horizontal")
    else:
        cax = divider.append_axes('right', size='5%', pad=0.1)
        fig.colorbar(im, cax=cax, ticks=ticks)

    color_w = 'white'
    color_b = 'black'

    # display less corr coefficients if too large (indicated by top position)
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            if not diagonal or j < i:
                if position == 'top' and large:
                    # if you want less precision in larger plots, set decimals=1
                    val = np.around(corr.iloc[i, j], decimals=2)
                else:
                    val = np.around(corr.iloc[i, j], decimals=2)
                color = color_b
                if abs(val) >= 0.6:
                    # on dark background should be written in white
                    color = color_w

                ax.text(j, i, val, ha='center', va='center',
                        color=color, fontsize=16)

    plt.tight_layout()

    save_path = f'img/{mode}/{crowdworker_group}/{title.replace(" ", "-")}'

    if is_p:
        save_path += '_p-values'

    plt.savefig(save_path)
    # plt.show()
    plt.close(fig)


def print_interesting_corr(corr, title):
    # helper method to find interesting correlations in terminal

    high = 0.7
    moderate = 0.5
    low = 0.3

    high_corr = []
    mod_corr = []
    low_corr = []
    high_neg_corr = []
    mod_neg_corr = []
    low_neg_corr = []

    for r in range(len(corr.columns)):
        for c in range(len(corr.columns)):
            if c > r:
                val = np.around(corr.iloc[r, c], decimals=1)
                if val >= high:
                    high_corr.append(
                        corr.columns[c] + " : " + corr.columns[r] + " (" + str(val) + ")")
                elif val >= moderate:
                    mod_corr.append(
                        corr.columns[c] + " : " + corr.columns[r] + " (" + str(val) + ")")
                elif val >= low:
                    low_corr.append(
                        corr.columns[c] + " : " + corr.columns[r] + " (" + str(val) + ")")
                elif abs(val) >= high:
                    high_neg_corr.append(
                        corr.columns[c] + " : " + corr.columns[r] + " (" + str(val) + ")")
                elif abs(val) >= moderate:
                    mod_neg_corr.append(
                        corr.columns[c] + " : " + corr.columns[r] + " (" + str(val) + ")")
                elif abs(val) >= low:
                    low_neg_corr.append(
                        corr.columns[c] + " : " + corr.columns[r] + " (" + str(val) + ")")

    for item in high_corr:
        print("High Correlation: %s\n" % item)
    for item in mod_corr:
        print("Moderate Correlation: %s\n" % item)
    for item in low_corr:
        print("Low Correlation: %s\n" % item)
    for item in high_neg_corr:
        print("High Neg. Correlation: %s\n" % item)
    for item in mod_neg_corr:
        print("Moderate Neg. Correlation: %s\n" % item)
    for item in low_neg_corr:
        print("Low Neg. Correlation: %s\n" % item)


def calculate_pvalues(df):
    df = df.dropna()._get_numeric_data()
    dfcols = pd.DataFrame(columns=df.columns)
    pvalues = dfcols.transpose().join(dfcols, how='outer')
    for r in df.columns:
        for c in df.columns:
            pvalues[r][c] = round(spearmanr(df[r], df[c])[1], 4)
    return pvalues


def print_inner_correlations():
    for title, df in inner_dimensional_dfs.items():
        corr = df.corr(method="spearman")
        print_interesting_corr(corr, title)

        p_values = calculate_pvalues(df)
        print(p_values)


def print_expert_correlations():
    corr_leaning_total = pd.DataFrame()
    leaning_total_sentences = pd.DataFrame()

    for title, df in leaning_dfs.items():
        corr = df.corr(method="spearman")
        last_row = corr.tail(1)
        row_inner = last_row.iloc[:, :-1]
        if "Article" in title:
            corr_leaning_total = corr_leaning_total.append(row_inner)
        else:
            leaning_total_sentences = leaning_total_sentences.append(row_inner)
        print_interesting_corr(corr, title)

        p_values = calculate_pvalues(df)

        print(p_values)

    print(corr_leaning_total)
    print(leaning_total_sentences)


def show_inner_correlations():
    for title, df in inner_dimensional_dfs.items():
        corr = df.corr(method="spearman")
        show_correlations(corr=corr,
                          title=title, large=True, diagonal=True, position='right', is_p=False)

        p_values = calculate_pvalues(df)
        show_correlations(corr=p_values,
                          title=title, large=True, diagonal=True, position='right', is_p=True)


def show_expert_correlations():
    corr_leaning_total = pd.DataFrame()
    leaning_total_sentences = pd.DataFrame()

    for title, df in leaning_dfs.items():
        corr = df.corr(method="spearman")
        last_row = corr.tail(1)
        row_inner = last_row.iloc[:, :-1]
        if "Article" in title:
            corr_leaning_total = corr_leaning_total.append(row_inner)
        else:
            leaning_total_sentences = leaning_total_sentences.append(row_inner)
        show_correlations(corr=corr, title=title, large=True,
                          diagonal=True, position='right', is_p=False)

        p_values = calculate_pvalues(df)

        show_correlations(corr=p_values,
                          title=title, large=True, diagonal=True, position='right', is_p=True)

    show_correlations(corr=leaning_total_sentences, title="Sentence Labels and Article Leanings", large=True,
                      diagonal=False, position='top', is_p=False)
    show_correlations(corr=corr_leaning_total, title="Article Labels and Article Leanings", large=True,
                      diagonal=False, position='top', is_p=False)


print_inner_correlations()
print_expert_correlations()
show_inner_correlations()
show_expert_correlations()
