import xlearn as xl

ffm_model = xl.create_ffm()


train_path = '/Users/admin/Downloads/xlearn-master/demo/classification/criteo_ctr/small_train.txt'
train_path2 = '/Users/admin/Downloads/libffm_toy/criteo.tr.r100.gbdt0.ffm'
test_path = '/Users/admin/Downloads/xlearn-master/demo/classification/criteo_ctr/small_test.txt'
test_path2 = '/Users/admin/Downloads/libffm_toy/criteo.va.r100.gbdt0.ffm'


ffm_model.setTrain(train_path2)
ffm_model.setValidate(test_path2)

# Parameters:
param = {'task':'reg', 
         'epoch': 3,
         'lr':0.2, 
         'lambda':0.002}

# Start to train
# The trained model will be stored in model.out
ffm_model.fit(param, './model.out')
ffm_model.setTXTModel('/Users/admin/Documents/github-workspace/recommender-system/factorization-machine/mdl.txt')
