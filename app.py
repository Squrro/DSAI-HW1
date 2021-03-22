
# You can write code above the if-main block.
if __name__ == '__main__':
    # You should not modify this part, but additional arguments are allowed.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--training',
                       default='training_data.csv',
                       help='input training data file name')

    parser.add_argument('--output',
                        default='submission.csv',
                        help='output file name')
    args = parser.parse_args()

    # The following part is an example.
    # You can modify it at will.
    import pandas as pd
    import math
    import pandas_datareader as web
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, RepeatVector
    from keras.layers.normalization import BatchNormalization
    from keras.optimizers import Adam
    from keras.callbacks import EarlyStopping, ModelCheckpoint


    df = pd.read_csv('./daily_data.csv')

    data = df.filter(['capacity'])
    dataset = data.values
    train_data_len = math.ceil(len(dataset))

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)

    train_data = scaled_data[0:train_data_len,:]
    x_train = []
    y_train = []
    for i in range(73,len(train_data)):
        x_train.append(train_data[i-73:i,0])
        y_train.append(train_data[i:0])
    
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1))
    
    model = Sequential()
    model.add(LSTM(50,return_sequences = True,input_shape=(x_train.shape[1],1)))
    model.add(LSTM(50,return_sequences = False))
    model.add(Dense(25))
    model.add(Dense(1))
    

    model.compile(optimizer='adam', loss = 'mean_squared_error')
    
    model.fit(x_train,y_train,batch_size=70,epochs=1)

    test_data = scaled_data[train_data_len - 73, :]
    x_test = []
    y_test = dataset[train_data_len:,:]
    for i in range(70,len(test_data)):
        x_test.append(test_data[i-73:i,0])
    
    x_test = np.array(x_test)

    x_test = np.reshape(x_test, (x_test.shape[0],73,1))
    
    prediction = model.predict(x_train)
    prediction = scaler.inverse_transform(prediction)

    import csv

    with open(args.output,'w',newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['date','operating_reserve(HW)'])
        writer.writerow(['20210323',int(prediction[0]*10)])
        writer.writerow(['20210324',int(prediction[1]*10)])
        writer.writerow(['20210325',int(prediction[2]*10)])
        writer.writerow(['20210326',int(prediction[3]*10)])
        writer.writerow(['20210327',int(prediction[4]*10)])
        writer.writerow(['20210328',int(prediction[5]*10)])
        writer.writerow(['20210329',int(prediction[6]*10)])

#prediction