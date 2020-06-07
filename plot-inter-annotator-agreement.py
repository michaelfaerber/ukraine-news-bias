import articleHandler as handler
from statistics import mean, median
from nltk.metrics import agreement, interval_distance
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import matplotlib as mpl
import matplotlib.font_manager as font_manager
import numpy

# calculation of agreement is only possible with all crowd workers, 
# ...since Krippendorff's alpha requires a constant amount of ratings
articles, workers = handler.get_articles("default")

jobs = ("subj", "hidden-assumpt", "framing", "bias")
govments = ('russia', 'ukraine', 'west')

categories = {
    'subj': set([2.0, 1.0, -1.0, -2.0]),
    'hidden-assumpt': set([0.0, 1.0, 2.0, 3.0]),
    'framing': set([2.0, 1.0, 0.0, -1.0, -2.0]),
    'bias': set([0.0, 1.0, 2.0, 3.0]),
}

categories_binary = {
    'subj': set([0.0, 1.0]),
    'hidden-assumpt': set([0.0, 1.0]),
    'framing': set([1.0, 0.0, -1.0]),
    'bias': set([0.0, 1.0]),
}


def get_rating_per_person_and_article(articles, person_index, type):
    rating_per_article = []

    def helper(x, job):
        if job != "framing":
            value = 0.0 if x < 2.0 else 1.0
        else:
            if x != 0.0:
                value = -1.0 if x < 0.0 else 1.0
            else:
                value = 0.0
        return value

    # map all values to 1.0 that are not 0.0
    def helper_soft(x, job):
        if job != "framing":
            value = 0.0 if x < 1.0 else 1.0
        else:
            if x != 0.0:
                value = -1.0 if x < 0.0 else 1.0
            else:
                value = 0.0
        return value

    if type != "default":
        mapping_func = helper if type == "binary" else helper_soft
    else:
        def mapping_func(x, y): return x

    for article in articles:
        rating = {}
        rating['id'] = article.article_id

        rating['subj'] = [mapping_func(sent.values['subj'][person_index], 'subj')
                          for sent in article.sentences]
        rating['hidden-assumpt'] = [mapping_func(sent.values['hidden-assumpt'][person_index], 'hidden-assumpt')
                                    for sent in article.sentences]
        rating['framing'] = [mapping_func(sent.values['framing'][gov][person_index], 'framing')
                             for gov in govments for sent in article.sentences]
        rating['bias'] = [mapping_func(sent.values['bias'][side][person_index], 'bias')
                          for side in ['pro-russia', 'pro-west'] for sent in article.sentences]

        rating_per_article.append(rating)
    return rating_per_article


def get_taskdata(art_index, all_raters, job):
    my_data = []

    for index, rater in enumerate(all_raters):
        name = 'rater' + str(index+1)
        my_data += [(name, 'item' + str(i), rater[art_index][job][i])
                    for i in range(len(rater[art_index][job]))]

    return my_data


def print_agreement_per_job_per_article(raters, job, noprint):
    all_agreement_scores = []  # contains agreement obj with id, fleiss and alpha score

    for art_index, article in enumerate(articles):

        # sequence of 3-tuples, each representing a coder's labeling of an item
        taskdata = get_taskdata(art_index, raters, job)
        article_agreement = get_article_agreement_helper(
            taskdata, categories[job], article.article_id)
        article_agreement["is_neutral"] = article_is_totally_neutral(
            article, job)
        all_agreement_scores.append(article_agreement)
        
    high_agreement = [
        article_agreement for article_agreement in all_agreement_scores if article_agreement['krippendorff'] == 1.0]
    neutral_art = [art for art in all_agreement_scores if art["is_neutral"]]
    neutral_within_high = [art for art in high_agreement if art["is_neutral"]]

    if not noprint:
        print("----")
        print(job)
        print(
            f"Amount of Articles with High Agreement: {len(high_agreement)} / 90 ( {len(high_agreement) / 90 * 100}%)")
        print(
            f"Amount of Articles Labeled as Neutral: {len(neutral_art)} / 90 ")
        print(
            f"Amount of Articles with High Agreement Labeled as Neutral: {len(neutral_within_high)} / {len(high_agreement)}")
        print(
            f"Median of All Articles: { median( [agr['krippendorff'] for agr in all_agreement_scores] )}")
        print(
            f"Upper Quartile of All Articles: { numpy.quantile( [agr['krippendorff'] for agr in all_agreement_scores], 0.75 )}")
        print(
            f"Lower Quartile of All Articles: { numpy.quantile( [agr['krippendorff'] for agr in all_agreement_scores], 0.25 )}")
    return all_agreement_scores


def print_krippendorff_per_job(raters, category):
    all_agreement_scores = []  # contains agreement obj with job and value

    for job in jobs:
        job_taskdata = []

        for i, rater in enumerate(raters):
            name = 'rater' + str(i)
            my_triplets = []

            for j, article_rating in enumerate(rater):
                item = 'item_' + str(j) + '_'
                my_article_triplets = [(name, item + str(k), rating)
                                       for k, rating in enumerate(article_rating[job])]
                my_triplets += my_article_triplets

            job_taskdata += my_triplets

        ratingtask = agreement.AnnotationTask(
            data=job_taskdata, distance=interval_distance)
        ratingtask.K = category[job]
        all_agreement_scores.append({job: ratingtask.alpha()})

    print('Krippendorff:')
    print(all_agreement_scores)

    return all_agreement_scores


def print_krippendorff_per_leaning(job, raters_normal):
    all_agreement_scores = []  # contains agreement obj with job, leaning and value

    for leaning in [-1, 0, 1]:
        leaning_taskdata = []
        article_id_current_leaning = [
            article.article_id for article in articles if int(article.leaning) == leaning]
        for i, rater in enumerate(raters_normal):
            name = 'rater' + str(i)
            my_triplets = []

            for j, article_rating in enumerate(rater):
                item = 'item_' + str(j) + '_'

                if article_rating['id'] in article_id_current_leaning:
                    my_article_triplets = [(name, item + str(k), rating)
                                           for k, rating in enumerate(article_rating[job])]
                    my_triplets += my_article_triplets

            leaning_taskdata += my_triplets

        ratingtask = agreement.AnnotationTask(
            data=leaning_taskdata, distance=interval_distance)
        ratingtask.K = categories[job]
        all_agreement_scores.append({leaning: ratingtask.alpha()})

    print('Krippendorff ' + job + ':')
    print(all_agreement_scores)

    return all_agreement_scores


def get_article_agreement_helper(my_data, my_cat, article_id):
    ratingtask = agreement.AnnotationTask(
        data=my_data, distance=interval_distance)
    ratingtask.K = my_cat

    article_agreement = {}
    article_agreement['id'] = article_id

    try:
        article_agreement['krippendorff'] = ratingtask.alpha()

    except ZeroDivisionError:
        article_agreement['krippendorff'] = 1.0

    return article_agreement


# contains array for each rater with rating objects for each article
raters_normal = [get_rating_per_person_and_article(
    articles, i, "default") for i in range(5)]

# contains article agreements for each job
articles_agreement = {}  

legend_elements = [Line2D([0], [0], marker='o', color='w', label='Label = 0 (neutral)', markerfacecolor='darkblue'), Line2D(
    [0], [0], marker='o', color='w', label='Label != 0', markerfacecolor='dodgerblue')]


def article_is_totally_neutral(article, job):
    if job == 'framing':
        # on article level, amount of neutral sentences are stored as tuple with 1.0 representing 100% (all) sentences
        val = int(article.scores[job]['russia']['avg'][1][1] != 1) + int(
            article.scores[job]['ukraine']['avg'][1][1] != 1) + int(article.scores[job]['west']['avg'][1][1] != 1)
    elif job == 'bias':
        val = int(article.scores[job]['pro-russia']['avg'] != 0) + int(
            article.scores[job]['pro-west']['avg'] != 0)
    else:
        val = article.scores[job]['avg']

    return val == 0


def show_krippendorff_per_article(raters):
    plt.figure(figsize=(11.0, 9.0))

    for i, job in enumerate(jobs):

        articles_agreement[job] = print_agreement_per_job_per_article(
            raters, job, noprint=True)

        my_ax = plt.subplot(4, 1, i+1)
        my_ax.grid(axis='y', linestyle='--')
        my_ax.set_title('Dimension: ' + job, fontname='CMU serif')

        if i == 2:
            my_ax.set_xlabel('Articles', fontname='CMU serif')
        my_ax.set_ylabel('Krippendorff\'s Alpha', fontname='CMU serif')

        my_ax.set_ylim(-1.5, 1.5)

        y_axis_data = [art['krippendorff'] for art in articles_agreement[job]]
        y_axis_data = sorted(y_axis_data)

        #print("Average article-wise " + job + ": ")
        #print(mean(y_axis_data))

        y_axis_data.reverse()

        color_per_value = []
        for art in articles_agreement[job]:
            color = 'darkblue' if art["is_neutral"] else 'dodgerblue'
            color_per_value.append(color)

        my_ax.scatter(x=range(1, 91), y=y_axis_data,
                      c=color_per_value, marker='x')

        my_ax.legend(handles=legend_elements,
                     loc="upper right", title="Article Label")

    plt.tight_layout()
    #plt.show()
    plt.savefig('img/annotator-agreement/krippendorff-per-article')

def print_sent_total_agreement(job):

    def get_values(sentence, job):
        values = []
        if job == "framing" or job == "bias":
            for key in sentence.values[job]:
                values.append(sentence.values[job][key])
        else:
            values.append(sentence.values[job])

        return values

    def has_total_agreement(val_arr):
        agreed = True
        for elems in val_arr:
            agreed = agreed and len(set(elems)) <= 1
        return agreed

    def not_neutral(arr):
        # if at least one arr is not neutral, it is set to NOT_NEUTRAL
        not_neutral = False
        for a in arr:
            not_neutral = not_neutral or 0.0 not in a
        return not_neutral

    sentences_values = [get_values(sent, job) for art in articles for sent in art.sentences]
    totally_agreed = [arr for arr in sentences_values if has_total_agreement(arr) ]
    not_neutral = [arr for arr in totally_agreed if not_neutral(arr)]
    print(f"Amount of Total Agreement of Sentences for {job}: {len(totally_agreed)} / 2057 ({len(totally_agreed)/2057 * 100}%)")
    print(f"Amount of Total Agreement of Not-Neutral Sentences for {job}: {len(not_neutral)} / 2057 ({len(not_neutral)/2057 * 100}%)")

    


show_krippendorff_per_article(raters_normal)

for job in jobs:
    print_krippendorff_per_leaning(job, raters_normal)
    print_agreement_per_job_per_article(raters_normal, job, False)
    print_sent_total_agreement(job)
    