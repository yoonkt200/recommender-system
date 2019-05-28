import xlearn as xl

if __name__ == "__main__":
    fm_model = xl.create_fm()

    train_path = '../dataset/train.txt'
    test_path = '../dataset/test.txt'

    fm_model.setTrain(train_path)
    fm_model.setValidate(test_path)

    # Parameters:
    param = {'task':'binary',
             'epoch': 10,
             'lr':0.2,
             'lambda':0.002,
             'metric': 'auc'}

    # Start to train
    # The trained model will be stored in model.out
    fm_model.fit(param, './model.out')
    fm_model.setTXTModel('./model.txt')

    # Prediction task
    fm_model.setTest(test_path)  # Set the path of test dataset
    fm_model.setSigmoid()                 # Convert output to 0-1

    # Start to predict
    # The output result will be stored in output.txt
    fm_model.predict("./model.out", "./output.txt")