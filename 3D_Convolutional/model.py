from tensorflow import keras
from time import time
st=time()
model=keras.Sequential()
model.add(   keras.layers.Conv3D(64,3,input_shape=(30, 192, 192, 1) , activation='relu')  )
model.add(   keras.layers.MaxPooling3D() )

model.add(   keras.layers.Conv3D(128,3, activation='relu')  )
model.add(   keras.layers.MaxPooling3D() )

model.add(   keras.layers.Conv3D(256,3 , activation='relu')  )
model.add(   keras.layers.MaxPooling3D() )

model.add(  keras.layers.TimeDistributed(keras.layers.Flatten()) )

model.add(  keras.layers.Bidirectional(keras.layers. GRU(128,return_sequences=True))  )
model.add(  keras.layers.Dropout(0.3))

model.add(  keras.layers.Bidirectional(keras.layers.GRU(128)) )
model.add(  keras.layers.Dropout(0.3))
model.add(  keras.layers.Dense(7,activation='softmax') )
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

print(model.summary())
model.save("raw_model.h5")
print(time()-st)