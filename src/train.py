import time
import hpfrec

def train():
    print('Training HPF model')
    recommender = hpfrec.HPF(k=latent_factors, random_seed=123,
                             ncores=-1, stop_crit='train-llk', verbose=True,
                             reindex=False, stop_thr=0.000001, maxiter=3000)
    recommender.step_size = None
    _logger.warning("Model is training, Don't interrupt.")
    recommender.fit(train_df)
    return recommender
    
if __name__ == "__main__":
    startTime = time.time()
    print('Training started at', startTime)
    train()
    print('Training took', int(time.time() - startTime), 'seconds')