import numpy as np
from scipy import stats
def get_sample_loss(xs, model):
    """Get the reconstruction loss of a given sample

    Args:
        xs (List[ModelInput]): A sample of data of shape that the autoencoder was trained on
        model (Autoencoder): Autoencoder model

    Returns:
        List[float]: returns the reconstruction losses of the elements with preserved indices
    """

    if len(xs) == 0:
        return []
    predict = model.predict(xs)
    predict = predict.reshape(predict.shape[0], predict.shape[2])
    xs_out = xs.reshape(xs.shape[0], xs.shape[2])
    return np.mean(np.abs(predict - xs_out), axis=1)

def z_score_loss_threshold(variable,loss, threshold):
    """Returns the set of values above and below the threshold

    Args:
        variable (List): The variable to iterate over
        loss (List): The loss to threshold by zscore
        threshold (float): The ZScore threshold

    Returns:
        above(list), below(list): A list of values from variable and np.nan. If np.isnan(x) then x is not above or below respectively
    """
    above = np.full(variable.shape, np.nan)
    below = np.full(variable.shape, np.nan)
    for idx, (val, score) in enumerate(zip(variable, stats.zscore(loss, axis=None))):
            above[idx] = np.where(score >= threshold, val, np.nan)
            below[idx] = np.where(score < threshold, val, np.nan)
    
    return above, below

def find_zscore_anomalies(variable, loss, threshold):
    above, _ = z_score_loss_threshold(variable,loss, threshold)
    # 1 if anomaly (in the set of `above`) 0 otherwise
    return list(map(lambda e: 0 if np.isnan(e[0,0]) and np.isnan(e[0,1]) else 1, above))