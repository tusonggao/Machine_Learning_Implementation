import time
from surprise import SVD, SlopeOne, KNNBasic
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import train_test_split

start_t_global = time.time()

# Load the movielens-100k dataset (download it if needed),
data = Dataset.load_builtin('ml-1m')
# data = Dataset.load_builtin('ml-100k')

# sample random trainset and testset
# test set is made of 25% of the ratings.
trainset, testset = train_test_split(data, test_size=.25, random_state=2019)

print('algo SVD: ')
start_t = time.time()
algo = SVD()
algo.fit(trainset)
predictions = algo.test(testset)
accuracy.rmse(predictions)
accuracy.mae(predictions)
print('SVD algo: cost time: ', time.time()-start_t)


print('KNNBasic algo:')
start_t = time.time()
algo = KNNBasic()
algo.fit(trainset)
predictions = algo.test(testset)
accuracy.rmse(predictions)
accuracy.mae(predictions)
print('KNNBasic algo: cost time: ', time.time()-start_t)


print('algo SlopeOne: ')
start_t = time.time()
algo = SlopeOne()
algo.fit(trainset)
predictions = algo.test(testset)
# Then compute RMSE
accuracy.rmse(predictions)
accuracy.mae(predictions)
print('algo SlopeOne: : cost time: ', time.time()-start_t)


print('prog ends here, cross_validate cost time', time.time()-start_t_global)