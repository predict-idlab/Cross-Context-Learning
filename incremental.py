import csv
from transfer import *


def transfer_from_to(origin, dest, best=True, incremental=False, repo=""):
    """
    Perform a transfer from one context to another.
    :param origin: origin context
    :param dest: destination context
    :param best: whether the origin context has the best transfer candidate or the worst
    :param incremental: not supported
    :param repo: where to store the results
    :return:
    """
    origin_node = int(origin.split("_")[-1])
    origin_city = origin.split("_")[0]
    dest_node = int(dest.split("_")[-1])
    dest_city = dest.split("_")[0]

    lengths = []
    ivals = []

    print("Train on original context", origin)
    train_or, model, anoms, ival1, mae_or, aucs_or, auc_avg_pr_or, fl_or = train_on_context(origin_node, origin_city,
                                                                                            override=True)  # original training
    print("Transfer model to target context", dest)
    train_dest, model, anoms, ival2a, mae_tr, aucs_tr, auc_avg_pr_tr, fl_tr = train_on_context(dest_node, dest_city,
                                                                                               model,
                                                                                               limited=True)  # transfer (without further training)

    lengths.extend([len(ival1.results_batch)])
    ivals.extend([ival1])

    if incremental:
        print("Incrementally train transferred model on target context", dest, "...")
        print("... with only the most recent fraction of data samples")
        last_ind = get_indices(train_or, train_dest, incremental_method='last')
        _, model, anoms, ival2d, mae_incr3, aucs_incr3, auc_avg_pr_incr3, fl_incr3 = train_on_context(dest_node,
                                                                                                      dest_city, model,
                                                                                                      incremental=True,
                                                                                                      limited=True,
                                                                                                      indices=None)  # transfer + incremental

        lengths.extend([len(ival2d.results_batch)])
        ivals.extend([ival2d])

    # for comparison
    print("Train from scratch on target context", dest)
    _, model, anoms, ival3, mae_dest1, aucs_dest1, auc_avg_pr_dest1, fl_dest1 = train_on_context(dest_node, dest_city,
                                                                                                 limited=True)
    print("Train from scratch on target context", dest)
    _, model, anoms, ival4, mae_dest2, aucs_dest2, auc_avg_pr_dest2, fl_dest2 = train_on_context(dest_node, dest_city,
                                                                                                 limited=False)

    lengths.extend([len(ival3.results_batch), len(ival4.results_batch)])
    ivals.extend([ival3, ival4])

    max_len = max(lengths)
    for ival in ivals:
        if len(ival.results_batch) > 0:
            ival.results_batch.extend([ival.results_batch[-1]] * (max_len - len(ival.results_batch)))

    outfile = repo
    if best:
        outfile += str(origin) + "_to_" + str(dest) + "_best.csv"
    else:
        outfile += str(origin) + "_to_" + str(dest) + "_worst.csv"

    with open(outfile, 'w') as csvfile:
        print("Writing to file", outfile)
        wr = csv.writer(csvfile, delimiter=',')
        wr.writerow(["node", "auc_score", "auc_avg_pr", "final loss"])
        wr.writerow(["transfer", str(aucs_tr), str(auc_avg_pr_tr), str(fl_tr)])
        if incremental:
            wr.writerow(["incremental - Last", str(aucs_incr3), str(auc_avg_pr_incr3), str(fl_incr3)])
        wr.writerow(["scratch - limited", str(aucs_dest1), str(auc_avg_pr_dest1), str(fl_dest1)])
        wr.writerow(["scratch - full", str(aucs_dest2), str(auc_avg_pr_dest2), str(fl_dest2)])

    return anoms, aucs_tr, aucs_dest1
