import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_score, auc, accuracy_score, recall_score, f1_score, matthews_corrcoef, \
    precision_recall_curve


# def poltting(test_labels, score, round, times, acc, k, m):
def poltting(test_labels, score, round, times, acc):
    auc_plt = []
    aupr_plt = []

    # average performance
    auc_list = []
    aupr_list = []
    f1_list = []
    precision_list = []
    recall_list = []
    mcc_list = []
    acc_list = []


   

    for i in range(times):
        auc_inner = []
        aupr_inner = []
        fpr, tpr, threshold = roc_curve(test_labels[i], score[i])
        auc_area = auc(fpr, tpr)
        precision, recall, _thresholds = precision_recall_curve(test_labels[i], score[i])
        aupr_area = auc(recall, precision)
        auc_inner.append(fpr)
        auc_inner.append(tpr)
        auc_inner.append(auc_area)
        aupr_inner.append(precision)
        aupr_inner.append(recall)
        aupr_inner.append(aupr_area)
        auc_plt.append(auc_inner)
        aupr_plt.append(aupr_inner)
        score_lables = [0 if j < 0.5 else 1 for j in score[i]]
        target = mutile_scores(test_labels[i], score_lables)
        print('auc_area:', "{:.4f}".format(auc_area), ' aupr_area:', "{:.4f}".format(aupr_area), ' F1:',
              "{:.4f}".format(target[0]), ' MCC:', "{:.4f}".format(target[1]), ' ACC:', "{:.4f}".format(acc[i]),
              ' Precision:', "{:.4f}".format(target[2]), ' RECALL:', "{:.4f}".format(target[3]))

       

        auc_list.append(auc_area)
        aupr_list.append(aupr_area)
        f1_list.append(target[0])
        precision_list.append(target[2])
        recall_list.append(target[3])
        mcc_list.append(target[1])
        acc_list.append(acc[i])

    auc_mean = sum(auc_list) / len(auc_list)
    aupr_mean = sum(aupr_list) / len(aupr_list)
    f1_mean = sum(f1_list) / len(f1_list)
    precision_mean = sum(precision_list) / len(precision_list)
    recall_mean = sum(recall_list) / len(recall_list)
    mcc_mean = sum(mcc_list) / len(mcc_list)
    acc_mean = sum(acc_list) / len(acc_list)

    with open(result_file_path,'a') as f:
        f.write('Round{} Average performance: \n'.format(str(round)))
        f.write('auc_area:' + "{:.4f}".format(auc_mean) + ' aupr_area:' + "{:.4f}".format(aupr_mean) + ' F1:' +
                "{:.4f}".format(f1_mean) + ' MCC:' + "{:.4f}".format(mcc_mean) + ' ACC:' + "{:.4f}".format(acc_mean) +
                ' Precision:' + "{:.4f}".format(precision_mean) + ' RECALL:' + "{:.4f}".format(recall_mean))
        f.write('\n\n')


    pltt(auc_plt, round, times, 'auc')
    pltt(aupr_plt, round, times, 'aupr')


def pltt(area, round, times, plotting_type):
    lw = 2
    plt.figure(figsize=(8, 5))
    for i in range(times):
        plt.plot(area[i][0], area[i][1],
                 lw=lw, label='ROC curve' + str(i) + ' (area = %0.4f)' % area[i][2])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    if plotting_type == 'auc':
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
    else:
        plt.xlabel('recall')
        plt.ylabel('Precision')
    plt.title('5-flod CV')
    plt.legend(loc="lower right")
    plt.savefig('../dataset/result/fig/fig round_' + str(round) + '_' + plotting_type + '.png')
    # plt.show()


def mutile_scores(test_lables, score_lables):
    f1 = f1_score(test_lables, score_lables)
    mcc = matthews_corrcoef(test_lables, score_lables)
    p = precision_score(test_lables, score_lables)
    recall = recall_score(test_lables, score_lables)
    target = [f1, mcc, p, recall]
    return target
