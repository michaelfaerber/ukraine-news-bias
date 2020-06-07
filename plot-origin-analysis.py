import articleHandler as handler
from functools import reduce
from numpy import mean
from numpy import std
import matplotlib.pyplot as plt
import numpy


class AnnotationData:
    def __init__(self, articles, workers):
        self.articles = articles
        self.workers = workers


jobs = ("subj", "hidden-assumpt", "framing", "bias")

#mode = "maj"
mode = "avg"

all_articles, all_workers = handler.get_articles("default")

west_articles, west_workers = handler.get_articles("west")

rest_articles, rest_workers = handler.get_articles("rest")


country_groups = {
    "default": AnnotationData(all_articles, all_workers),
    "west": AnnotationData(west_articles, west_workers),
    "rest": AnnotationData(rest_articles, rest_workers)  # ,
}


def get_nbr_of_unrated_sentences(articles, job):
    counter = 0

    for a in articles:
        for sent in a.sentences:
            if job == "framing":
                if len(sent.values["framing"]["russia"]) == 0:
                    counter += 1
            elif job == "bias":
                if len(sent.values["bias"]["pro-russia"]) == 0:
                    counter += 1
            else:
                if len(sent.values[job]) == 0:
                    counter += 1

    return counter


def get_all_unrated_sentences_of_cwgroup(articles):
    amount = 0
    total = 0

    for job in jobs:
        total += 2057  # amount of total sentences
        this_amount = get_nbr_of_unrated_sentences(articles, job)
        amount += this_amount
        print(f"Unrated sentences of {job}: {this_amount} / 2057")

    print(f"Unrated sentences of cw group: {amount} / {total}")


def print_unrated_sentences_statistics():
    for groupname, annotation in country_groups.items():
        print(f"Cw Group: {groupname}")
        get_all_unrated_sentences_of_cwgroup(annotation.articles)
        print("-----")


def print_amount_of_workers_for_all_groups():

    for groupname, annotation in country_groups.items():
        count_all = reduce(
            lambda x, y: x+y, [len(arr) for arr in annotation.workers.values()])
        print(f"Amount of Workers in {groupname}: {count_all}")
        print("------")


def print_workers_per_country_in_group(groupname):

    worker_per_country = {}

    def mapWorkerToCountry(current_worker, country_list):
        if current_worker.country in country_list:
            country_list[current_worker.country] += 1
        else:
            country_list[current_worker.country] = 1

    # workers have one array for each bias dimension with worker objects
    for worker_arr in country_groups[groupname].workers.values():
        for worker in worker_arr:
            mapWorkerToCountry(worker, worker_per_country)

    print(f"Workers per Country: {worker_per_country} ")
    print("-----")

    return worker_per_country


def is_judged(sent, job):
    scores = sent.scores[job]

    if mode in scores:
        return scores[mode] is not None
    else:
        for key in scores:
            return scores[key][mode] is not None


def print_biased_sentences_in_group(groupname):
    articles = country_groups[groupname].articles
    sentences = [sent for art in articles for sent in art.sentences]

    print(groupname)

    for job in jobs:
        all_judged_sentences = [
            sent for sent in sentences if is_judged(sent, job)]
        count_judged = len(all_judged_sentences)
        count_biased = len(
            [sent for sent in all_judged_sentences if sent.is_biased(job, mode)])
        print(f"Total of judged sentences for {job}: {count_judged}")
        print(f"Total of biased sentences for {job}: {count_biased}")

        if count_judged != 0:
            print(
                f"Percentage of biased sentences for {job}: {(count_biased/count_judged) * 100} %")

            if job == "framing":
                count_pos = {}
                count_neg = {}

                for gov in ["russia", "west", "ukraine"]:
                    count_pos[gov] = len(
                        [sent for sent in all_judged_sentences if sent.scores["framing"][gov][mode] > 0])
                    count_neg[gov] = len(
                        [sent for sent in all_judged_sentences if sent.scores["framing"][gov][mode] < 0])

                    print(
                        f"Total of pos. sentences for {gov}: {count_pos[gov]}")
                    print(
                        f"Percentage of pos. sentences for {gov}: {(count_pos[gov]/count_judged) * 100} %")
                    print(
                        f"Total of neg. sentences for {gov}: {count_neg[gov]}")
                    print(
                        f"Percentage of neg. sentences for {gov}: {(count_neg[gov]/count_judged) * 100} %")

            if job == "bias":
                count_direction = {}
                for tendency in ["pro-russia", "pro-west"]:
                    count_direction[tendency] = len(
                        [sent for sent in all_judged_sentences if sent.scores["bias"][tendency][mode] > 1])

                    print(
                        f"Total of sentences for {tendency}: {count_direction[tendency]}")
                    print(
                        f"Percentage of sentences for {tendency}: {(count_direction[tendency]/count_judged) * 100} %")

        print("-----")


def print_intersection_biased_sent_west_rest():
    west_sentences = [
        sent for art in country_groups["west"].articles for sent in art.sentences]
    rest_sentences = [
        sent for art in country_groups["rest"].articles for sent in art.sentences]

    for job in jobs:
        judged_rest = [sent for sent in rest_sentences if is_judged(sent, job)]
        judged_west = [sent for sent in west_sentences if is_judged(sent, job)]
        biased_rest = [sent for sent in judged_rest if sent.is_biased(job, mode)]
        biased_west = [sent for sent in judged_west if sent.is_biased(job, mode)]

        west_content = [sent.content for sent in judged_west]
        west_content_biased = [sent.content for sent in biased_west]

        judgement_intersection = [
            sent for sent in judged_rest if sent.content in west_content]
        biased_intersection = [
            sent for sent in biased_rest if sent.content in west_content_biased]

        print(
            f"Intersection of Judged Sentences of {job}: {len(judgement_intersection)}")
        print(
            f"Intersection of Biased Sentences of {job}: {len(biased_intersection)}")

        if job == "framing":

            count_intersection_pos = {}
            count_intersection_neg = {}

            def sentenceIsFramedInBoth(sent_rest, gov, direction):
                isFramed = (lambda x: x < 0, lambda x: x > 0)[
                    direction == "pos"]
                sent_framed = isFramed(sent_rest.scores["framing"][gov][mode])
                sent_west = next(
                    (sent for sent in biased_west if sent.content == sent_rest.content), None)

                if sent_west is not None:
                    return sent_framed and isFramed(sent_west.scores["framing"][gov][mode])

                return False

            for gov in ["russia", "west", "ukraine"]:
                count_intersection_pos[gov] = len(
                    [sent for sent in biased_intersection if sentenceIsFramedInBoth(sent, gov, "pos")])
                count_intersection_neg[gov] = len(
                    [sent for sent in biased_intersection if sentenceIsFramedInBoth(sent, gov, "neg")])

                print(
                    f"Total of pos. sentences for {gov} in rest and west: {count_intersection_pos[gov]}")
                print(
                    f"Percentage of pos. sentences for {gov} in rest and west: {(count_intersection_pos[gov]/len(judgement_intersection)) * 100} %")
                print(
                    f"Total of neg. sentences for {gov} in rest and west: {count_intersection_neg[gov]}")
                print(
                    f"Percentage of neg. sentences for {gov}: {(count_intersection_neg[gov]/len(judgement_intersection)) * 100} %")

        if job == "bias":
            count_direction = {}

            def sentenceIsBiasedInBoth(sent_rest, tendency):
                biased_in_rest = sent_rest.scores["bias"][tendency][mode] > 1
                sent_west = next(
                    (sent for sent in biased_west if sent.content == sent_rest.content), None)

                if sent_west is not None:
                    biased_in_west = sent_west.scores["bias"][tendency][mode] > 1
                    return biased_in_rest and biased_in_west

                return False

            for tendency in ["pro-russia", "pro-west"]:
                count_direction[tendency] = len(
                    [sent for sent in biased_intersection if sentenceIsBiasedInBoth(sent, tendency)])

                print(
                    f"Total of sentences for {tendency}: {count_direction[tendency]}")
                print(
                    f"Percentage of sentences for {tendency}: {(count_direction[tendency]/len(judgement_intersection)) * 100} %")

        print("-----")


def get_avg_sent_values_in_group(groupname):
    articles = country_groups[groupname].articles
    sentences = [sent for art in articles for sent in art.sentences]

    print(groupname)

    avg_values = {}
    all_sent_values = {}

    for job in jobs:

        avg_values[job] = {}
        all_sent_values[job] = {}

        if job == "framing" or job == "bias":
            for tend in sentences[0].values[job]:
                all_sent_values[job][tend] = [
                    item for sent in sentences for item in sent.values[job][tend]]
        else:
            # add level in dictionary to easily iterate
            all_sent_values[job]["all"] = [
                item for sent in sentences for item in sent.values[job]]

        for level in all_sent_values[job]:
            if len(all_sent_values[job][level]) != 0:

                mean_val = round(mean(all_sent_values[job][level]), 2)
                std_dev = round(std(all_sent_values[job][level]), 2)

                avg_values[job][level] = {
                    "mean": mean_val,
                    "std": std_dev
                }

                print(
                    f"Average of judgements on {job}, {level}: {mean_val} ({std_dev})")

        print("-----")

    return avg_values, all_sent_values


def print_bias_tendency_per_expert(groupname):
    # is compared on article level!
    articles = country_groups[groupname].articles
    print(groupname)

    sentences_per_expert_label = {
        "neutral": [sent for art in articles for sent in art.sentences if art.leaning == 0 and is_judged(sent, "bias")],
        "pro-rus": [sent for art in articles for sent in art.sentences if art.leaning == -1 and is_judged(sent, "bias")],
        "pro-west": [sent for art in articles for sent in art.sentences if art.leaning == 1 and is_judged(sent, "bias")]
    }

    for key in sentences_per_expert_label:
        print(f"Sentences of Articles judged as {key} by expert:")

        count_bias_pro_russia = len(
            [sent for sent in sentences_per_expert_label[key] if sent.scores["bias"]["pro-russia"][mode] > 1])
        count_bias_pro_west = len(
            [sent for sent in sentences_per_expert_label[key] if sent.scores["bias"]["pro-west"][mode] > 1])

        print(
            f"Amount of sentences judged as pro-russia by crowdworkers: {count_bias_pro_russia}")
        print(
            f"Amount of sentences judged as pro-west by crowdworkers: {count_bias_pro_west}")
        print("-----")


def print_bias_tendency_per_expert_west_rest():
    rest_articles = country_groups["rest"].articles
    west_articles = country_groups["west"].articles

    sentences_per_expert_label = {
        "rest": {
            "neutral-e": [sent for art in rest_articles for sent in art.sentences if art.leaning == 0 and is_judged(sent, "bias")],
            "pro-rus-e": [sent for art in rest_articles for sent in art.sentences if art.leaning == -1 and is_judged(sent, "bias")],
            "pro-west-e": [sent for art in rest_articles for sent in art.sentences if art.leaning == 1 and is_judged(sent, "bias")]

        },
        "west": {
            "neutral-e": [sent for art in west_articles for sent in art.sentences if art.leaning == 0 and is_judged(sent, "bias")],
            "pro-rus-e": [sent for art in west_articles for sent in art.sentences if art.leaning == -1 and is_judged(sent, "bias")],
            "pro-west-e": [sent for art in west_articles for sent in art.sentences if art.leaning == 1 and is_judged(sent, "bias")]
        }
    }

    # find missing sentences
    missing_rest = [sent for art in rest_articles for sent in art.sentences if is_judged(
        sent, "bias") and (art.leaning not in [0, 1, -1])]
    missing_west = [sent for art in west_articles for sent in art.sentences if is_judged(
        sent, "bias") and (art.leaning not in [0, 1, -1])]

    # print amount of judged sentences
    for group in sentences_per_expert_label:
        for key in sentences_per_expert_label[group]:
            print(
                f"Amount of judged sentences by {group} rated as {key} by expert: {len(sentences_per_expert_label[group][key])}")

    for key in sentences_per_expert_label["rest"]:
        judged_west_content = [
            sent.content for sent in sentences_per_expert_label["west"][key]]
        judged_by_both = [sent for sent in sentences_per_expert_label["rest"]
                          [key] if sent.content in judged_west_content]
        print(
            f"Amount of sentences rated as {key} by expert and judged by both: {len(judged_by_both)}")
    print("-----")

    # print amount of biased sentences with tendency per expert category
    biased_sentences = {}
    for group in sentences_per_expert_label:
        biased_sentences[group] = {}

        for key in sentences_per_expert_label[group]:
            biased_sentences[group][key] = {}

            print(group)
            print(f"Sentences of Articles judged as {key} by expert:")

            bias_pro_russia = [sent for sent in sentences_per_expert_label[group]
                               [key] if sent.scores["bias"]["pro-russia"][mode] > 1]
            bias_pro_west = [sent for sent in sentences_per_expert_label[group]
                             [key] if sent.scores["bias"]["pro-west"][mode] > 1]

            biased_sentences[group][key]["pro-russia-cw"] = bias_pro_russia
            biased_sentences[group][key]["pro-west-cw"] = bias_pro_west

            print(
                f"Amount of sentences judged as pro-russia by crowdworkers: {len(bias_pro_russia)}")
            print(
                f"Amount of sentences judged as pro-west by crowdworkers: {len(bias_pro_west)}")
            print("-----")

    # find intersection
    print("West and Rest Intersection")

    for key in sentences_per_expert_label["rest"]:

        print(f"Sentences of Articles judged as {key} by expert:")

        for tendency in ["pro-russia-cw", "pro-west-cw"]:
            west_content = [
                sent.content for sent in biased_sentences["west"][key][tendency]]
            biased_in_both = [sent for sent in biased_sentences["rest"]
                              [key][tendency] if sent.content in west_content]
            print(
                f"Sentences judged as {tendency} by cw from both west and rest: {len(biased_in_both)}")

        print("-----")



def show_stacked_errorbar_all_in_one(west_vals, rest_vals):

    xticklabels = {
        "all": ["Subjectivity", "Hidden\nAssumptions", "Bias:\nPro-Russia", "Bias:\nPro-West"]
    }

    xticks = {
        "west": [0, 3, 6, 9],
        "rest": [1, 4, 7, 10]
    }

    my_ticks = []
    for i in numpy.arange(len(xticks["west"])):
        my_ticks.append(xticks["west"][i])
        my_ticks.append(xticks["rest"][i])

    ylim = {
        "all": [-0.5, 3.5]
    }

    xlim = {
        "all": (-1, 11)
    }

    title = {
        "all": "All Dimensions with 4-Point Scale"
    }

    yticklabels = {
        "all": ["not present (0)", "rather not present (1)", "rather present (2)", "present (3)"]
    }

    yticks = {
        "all": [0, 1, 2, 3]
    }

    fig, ax = plt.subplots(figsize=(7, 4))

    # ax.set_title(title["all"])
    ax.set_xticks(my_ticks, minor=True)
    ax.set_xticks([elem + 0.5 for elem in xticks["west"]], minor=False)
    ax.set_xticklabels(["" for elem in my_ticks], minor=True)
    ax.set_xticklabels(xticklabels["all"], minor=False)

    # do not show ticks of label position
    ax.tick_params(axis="x", which="major", length=0, width=0)

    ax.set_yticklabels(yticklabels["all"])
    ax.set_yticks(yticks["all"])
    plt.setp(ax.get_yticklabels(), ha="right",
             rotation=45, rotation_mode="anchor")

    ax.set_xlim(xlim["all"])
    ax.set_ylim(ylim["all"])
    ax.grid(b=True, ls=":", axis="y")

    for elem in xticks["rest"]:
        position = elem + 1
        ax.axvline(x=position, ls=":")

    means_west = []
    std_west = []
    means_rest = []
    std_rest = []

    for job in west_vals:
        if job != "framing":
            means_west += [west_vals[job][tend]["mean"]
                           for tend in west_vals[job]]
            std_west += [west_vals[job][tend]["std"]
                         for tend in west_vals[job]]
            means_rest += [rest_vals[job][tend]["mean"]
                           for tend in rest_vals[job]]
            std_rest += [rest_vals[job][tend]["std"]
                         for tend in rest_vals[job]]

    ax.errorbar(xticks["west"], means_west, std_west, fmt='_k',
                lw=4, ms=12, label="From West Countries")
    ax.errorbar(xticks["rest"], means_rest, std_rest, fmt='_k',
                lw=4, ms=12, label="From Other Countries", mec="b", ecolor="b")

    ax.legend(loc="upper right")

    points_west_and_rest = []

    counter = 0
    end = len(means_west)
    points_west_and_rest = []
    my_ticks = []

    while counter < end:
        points_west_and_rest.append(means_west[counter])
        points_west_and_rest.append(means_rest[counter])

        my_ticks.append(xticks["west"][counter])
        my_ticks.append(xticks["rest"][counter])

        counter += 1

    for i, j in zip(my_ticks, points_west_and_rest):
        # adds a little correction to put annotation in marker's centrum
        ax.annotate(str(j),  xy=(i + 0.22, j - 0.05), size="small")

    plt.tight_layout()
    # plt.show()

    plt.savefig(f".\\img\\west-rest-comparison\\subj-hiddenassumpt-bias")


def show_stacked_errorbar_framing(west_vals, rest_vals):

    xticklabels = {
        "all": ["Russian\nRepresentation", "Ukrainian\nRepresentation", "West\nRepresentation"]
    }

    xticks = {
        "west": [0, 3, 6],
        "rest": [1, 4, 7]
    }

    my_ticks = []
    for i in numpy.arange(len(xticks["west"])):
        my_ticks.append(xticks["west"][i])
        my_ticks.append(xticks["rest"][i])

    ylim = {
        "all": [-2.5, 2.5]
    }

    xlim = {
        "all": (-1, 8)
    }

    yticklabels = {
        "all": ["positive (-2)", "rather positive (-1)", "neutral (0)", "rather positive (1)", "positive (2)"]
    }

    yticks = {
        "all": [-2, -1, 0, 1, 2]
    }

    fig, ax = plt.subplots(figsize=(6, 4))

    # ax.set_title(title["all"])
    ax.set_xticks(my_ticks, minor=True)
    ax.set_xticks([elem + 0.5 for elem in xticks["west"]], minor=False)
    ax.set_xticklabels(["" for elem in my_ticks], minor=True)
    ax.set_xticklabels(xticklabels["all"], minor=False)

    # do not show ticks of label position
    ax.tick_params(axis="x", which="major", length=0, width=0)

    ax.set_yticklabels(yticklabels["all"])
    ax.set_yticks(yticks["all"])
    plt.setp(ax.get_yticklabels(), ha="right",
             rotation=45, rotation_mode="anchor")

    ax.set_xlim(xlim["all"])
    ax.set_ylim(ylim["all"])
    ax.grid(b=True, ls=":", axis="y")

    for elem in xticks["rest"]:
        position = elem + 1
        ax.axvline(x=position, ls=":")

    means_west = []
    std_west = []
    means_rest = []
    std_rest = []

    job = "framing"

    means_west += [west_vals[job][tend]["mean"] for tend in west_vals[job]]
    std_west += [west_vals[job][tend]["std"] for tend in west_vals[job]]
    means_rest += [rest_vals[job][tend]["mean"] for tend in rest_vals[job]]
    std_rest += [rest_vals[job][tend]["std"] for tend in rest_vals[job]]

    ax.errorbar(xticks["west"], means_west, std_west, fmt='_k',
                lw=4, ms=12, label="From West Countries")
    ax.errorbar(xticks["rest"], means_rest, std_rest, fmt='_k',
                lw=4, ms=12, label="From Other Countries", mec="b", ecolor="b")

    ax.legend(loc="upper right")

    points_west_and_rest = []

    counter = 0
    end = len(means_west)
    points_west_and_rest = []
    my_ticks = []

    while counter < end:
        points_west_and_rest.append(means_west[counter])
        points_west_and_rest.append(means_rest[counter])

        my_ticks.append(xticks["west"][counter])
        my_ticks.append(xticks["rest"][counter])

        counter += 1

    for i, j in zip(my_ticks, points_west_and_rest):
        # adds a little correction to put annotation in marker's centrum
        ax.annotate(str(j),  xy=(i + 0.22, j - 0.05), size="small")

    plt.tight_layout()
    # plt.show()

    plt.savefig(f".\\img\\west-rest-comparison\\framing")


# print_amount_of_workers_for_all_groups()
# print_workers_per_country_in_group("default")

# for key in ["west", "rest"]:
    # print_biased_sentences_in_group(key)
    # print_bias_tendency_per_expert(key)

# print_bias_tendency_per_expert_west_rest()
# print_intersection_biased_sent_west_rest()

avg_values_west, all_data_west = get_avg_sent_values_in_group("west")
avg_values_rest, all_data_rest = get_avg_sent_values_in_group("rest")

show_stacked_errorbar_all_in_one(avg_values_west, avg_values_rest)
show_stacked_errorbar_framing(avg_values_west, avg_values_rest)



