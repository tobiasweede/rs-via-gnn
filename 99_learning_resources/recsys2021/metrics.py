from sklearn.metrics import average_precision_score, log_loss


def calculate_ctr(gt):
    positive = len([x for x in gt if x == 1])
    ctr = positive / float(len(gt))
    return ctr


def compute_rce(pred, gt):
    cross_entropy = log_loss(gt, pred)
    data_ctr = calculate_ctr(gt)
    strawman_cross_entropy = log_loss(gt, [data_ctr for _ in range(len(gt))])
    return (1.0 - cross_entropy / strawman_cross_entropy) * 100.0


ground_truth = read_predictions("gt.csv")  # will return data in the form (tweet_id, user_id, labed (1 or 0))
predictions = read_predictions("predictions.csv")  # will return data in the form (tweet_id, user_id, prediction)

rce = compute_rce(predictions, ground_truth)
average_precision = average_precision_score(ground_truth, predictions)

Add