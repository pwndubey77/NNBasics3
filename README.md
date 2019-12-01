# validation accuracy base model 
  82.32

# Network-sources

model_depthSep.add(SeparableConv2D(20, kernel_size=(3, 3), activation='relu', input_shape= (32,32,3))) #output 30 , RF = 3x3
model_depthSep.add(BatchNormalization()) 
model_depthSep.add(Dropout(0.25))

model_depthSep.add(SeparableConv2D(40, kernel_size=(3, 3), activation='relu'))   #output 28 , RF = 5x5
model_depthSep.add(BatchNormalization())
model_depthSep.add(MaxPooling2D(pool_size=(2, 2))) #output 14 , RF = 6x6
model_depthSep.add(Dropout(0.25))

model_depthSep.add(SeparableConv2D(80,kernel_size=(3, 3), activation='relu')) #output 12 , RF = 10x10
model_depthSep.add(BatchNormalization())
model_depthSep.add(MaxPooling2D(pool_size=(2, 2))) #output 6 , RF = 12x12
model_depthSep.add(Dropout(0.25))

model_depthSep.add(SeparableConv2D(160, kernel_size=(3, 3), activation='relu')) #output 4 , RF = 20x20
model_depthSep.add(BatchNormalization())
model_depthSep.add(SeparableConv2D(320, kernel_size=(3, 3), activation='relu')) #output 2 , RF = 28x28
model_depthSep.add(BatchNormalization())
model_depthSep.add(Dropout(0.25))

model_depthSep.add(Flatten())

model_depthSep.add(Dropout(0.25))
model_depthSep.add(Dense(num_classes, activation='softmax'))

# Epoch logs
Epoch 1/50
195/195 [==============================] - 29s 148ms/step - loss: 0.3005 - acc: 0.8939 - val_loss: 0.2812 - val_acc: 0.8965
Epoch 2/50
195/195 [==============================] - 19s 95ms/step - loss: 0.2365 - acc: 0.9094 - val_loss: 0.2208 - val_acc: 0.9133
Epoch 3/50
195/195 [==============================] - 19s 95ms/step - loss: 0.2115 - acc: 0.9178 - val_loss: 0.2188 - val_acc: 0.9172
Epoch 4/50
195/195 [==============================] - 19s 97ms/step - loss: 0.1931 - acc: 0.9249 - val_loss: 0.1914 - val_acc: 0.9270
Epoch 5/50
195/195 [==============================] - 19s 96ms/step - loss: 0.1792 - acc: 0.9299 - val_loss: 0.1657 - val_acc: 0.9352
Epoch 6/50
195/195 [==============================] - 19s 97ms/step - loss: 0.1700 - acc: 0.9334 - val_loss: 0.1649 - val_acc: 0.9363
Epoch 7/50
195/195 [==============================] - 19s 96ms/step - loss: 0.1626 - acc: 0.9363 - val_loss: 0.1608 - val_acc: 0.9371
Epoch 8/50
195/195 [==============================] - 19s 96ms/step - loss: 0.1568 - acc: 0.9384 - val_loss: 0.1508 - val_acc: 0.9416
Epoch 9/50
195/195 [==============================] - 19s 96ms/step - loss: 0.1504 - acc: 0.9412 - val_loss: 0.1477 - val_acc: 0.9436
Epoch 10/50
195/195 [==============================] - 19s 96ms/step - loss: 0.1459 - acc: 0.9426 - val_loss: 0.1369 - val_acc: 0.9474
Epoch 11/50
195/195 [==============================] - 19s 96ms/step - loss: 0.1412 - acc: 0.9446 - val_loss: 0.1345 - val_acc: 0.9480
Epoch 12/50
195/195 [==============================] - 19s 96ms/step - loss: 0.1381 - acc: 0.9456 - val_loss: 0.1362 - val_acc: 0.9475
Epoch 13/50
195/195 [==============================] - 19s 95ms/step - loss: 0.1352 - acc: 0.9471 - val_loss: 0.1305 - val_acc: 0.9499
Epoch 14/50
195/195 [==============================] - 19s 96ms/step - loss: 0.1307 - acc: 0.9489 - val_loss: 0.1377 - val_acc: 0.9470
Epoch 15/50
195/195 [==============================] - 19s 97ms/step - loss: 0.1295 - acc: 0.9492 - val_loss: 0.1362 - val_acc: 0.9476
Epoch 16/50
195/195 [==============================] - 19s 96ms/step - loss: 0.1272 - acc: 0.9503 - val_loss: 0.1269 - val_acc: 0.9508
Epoch 17/50
195/195 [==============================] - 19s 96ms/step - loss: 0.1244 - acc: 0.9512 - val_loss: 0.1272 - val_acc: 0.9507
Epoch 18/50
195/195 [==============================] - 19s 97ms/step - loss: 0.1229 - acc: 0.9515 - val_loss: 0.1241 - val_acc: 0.9517
Epoch 19/50
195/195 [==============================] - 19s 96ms/step - loss: 0.1206 - acc: 0.9525 - val_loss: 0.1240 - val_acc: 0.9518
Epoch 20/50
195/195 [==============================] - 19s 96ms/step - loss: 0.1194 - acc: 0.9529 - val_loss: 0.1191 - val_acc: 0.9540
Epoch 21/50
195/195 [==============================] - 19s 95ms/step - loss: 0.1174 - acc: 0.9538 - val_loss: 0.1260 - val_acc: 0.9518
Epoch 22/50
195/195 [==============================] - 19s 95ms/step - loss: 0.1160 - acc: 0.9546 - val_loss: 0.1231 - val_acc: 0.9530
Epoch 23/50
195/195 [==============================] - 19s 96ms/step - loss: 0.1152 - acc: 0.9548 - val_loss: 0.1305 - val_acc: 0.9500
Epoch 24/50
195/195 [==============================] - 19s 96ms/step - loss: 0.1129 - acc: 0.9554 - val_loss: 0.1237 - val_acc: 0.9528
Epoch 25/50
195/195 [==============================] - 19s 95ms/step - loss: 0.1122 - acc: 0.9556 - val_loss: 0.1220 - val_acc: 0.9532
Epoch 26/50
195/195 [==============================] - 19s 96ms/step - loss: 0.1102 - acc: 0.9568 - val_loss: 0.1393 - val_acc: 0.9481
Epoch 27/50
195/195 [==============================] - 19s 95ms/step - loss: 0.1101 - acc: 0.9570 - val_loss: 0.1205 - val_acc: 0.9533
Epoch 28/50
195/195 [==============================] - 18s 95ms/step - loss: 0.1086 - acc: 0.9574 - val_loss: 0.1213 - val_acc: 0.9530
Epoch 29/50
195/195 [==============================] - 19s 95ms/step - loss: 0.1080 - acc: 0.9576 - val_loss: 0.1194 - val_acc: 0.9538
Epoch 30/50
195/195 [==============================] - 19s 95ms/step - loss: 0.1066 - acc: 0.9578 - val_loss: 0.1154 - val_acc: 0.9562
Epoch 31/50
195/195 [==============================] - 19s 95ms/step - loss: 0.1062 - acc: 0.9584 - val_loss: 0.1239 - val_acc: 0.9531
Epoch 32/50
195/195 [==============================] - 19s 96ms/step - loss: 0.1053 - acc: 0.9586 - val_loss: 0.1119 - val_acc: 0.9578
Epoch 33/50
195/195 [==============================] - 19s 95ms/step - loss: 0.1041 - acc: 0.9591 - val_loss: 0.1201 - val_acc: 0.9548
Epoch 34/50
195/195 [==============================] - 18s 95ms/step - loss: 0.1036 - acc: 0.9592 - val_loss: 0.1175 - val_acc: 0.9555
Epoch 35/50
195/195 [==============================] - 19s 95ms/step - loss: 0.1024 - acc: 0.9600 - val_loss: 0.1163 - val_acc: 0.9556
Epoch 36/50
195/195 [==============================] - 19s 95ms/step - loss: 0.1021 - acc: 0.9597 - val_loss: 0.1185 - val_acc: 0.9555
Epoch 37/50
195/195 [==============================] - 19s 95ms/step - loss: 0.1008 - acc: 0.9602 - val_loss: 0.1224 - val_acc: 0.9540
Epoch 38/50
195/195 [==============================] - 19s 96ms/step - loss: 0.1000 - acc: 0.9610 - val_loss: 0.1223 - val_acc: 0.9536
Epoch 39/50
195/195 [==============================] - 18s 94ms/step - loss: 0.0999 - acc: 0.9606 - val_loss: 0.1133 - val_acc: 0.9568
Epoch 40/50
195/195 [==============================] - 19s 95ms/step - loss: 0.0991 - acc: 0.9607 - val_loss: 0.1133 - val_acc: 0.9569
Epoch 41/50
195/195 [==============================] - 19s 95ms/step - loss: 0.0990 - acc: 0.9612 - val_loss: 0.1127 - val_acc: 0.9577
Epoch 42/50
195/195 [==============================] - 19s 95ms/step - loss: 0.0974 - acc: 0.9617 - val_loss: 0.1154 - val_acc: 0.9567
Epoch 43/50
195/195 [==============================] - 18s 94ms/step - loss: 0.0978 - acc: 0.9617 - val_loss: 0.1154 - val_acc: 0.9564
Epoch 44/50
195/195 [==============================] - 18s 94ms/step - loss: 0.0963 - acc: 0.9621 - val_loss: 0.1157 - val_acc: 0.9561
Epoch 45/50
195/195 [==============================] - 18s 95ms/step - loss: 0.0958 - acc: 0.9624 - val_loss: 0.1254 - val_acc: 0.9527
Epoch 46/50
195/195 [==============================] - 18s 95ms/step - loss: 0.0959 - acc: 0.9622 - val_loss: 0.1131 - val_acc: 0.9565
Epoch 47/50
195/195 [==============================] - 19s 95ms/step - loss: 0.0944 - acc: 0.9629 - val_loss: 0.1120 - val_acc: 0.9572
Epoch 48/50
195/195 [==============================] - 19s 96ms/step - loss: 0.0945 - acc: 0.9629 - val_loss: 0.1126 - val_acc: 0.9573
Epoch 49/50
195/195 [==============================] - 19s 95ms/step - loss: 0.0940 - acc: 0.9630 - val_loss: 0.1182 - val_acc: 0.9555
Epoch 50/50
195/195 [==============================] - 19s 95ms/step - loss: 0.0924 - acc: 0.9638 - val_loss: 0.1109 - val_acc: 0.9584
