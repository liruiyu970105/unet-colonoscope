from data import *
from model import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"



data_gen_args = dict(rotation_range=0.2,
                     width_shift_range=0.05,
                     height_shift_range=0.05,
                     shear_range=0.05,
                     zoom_range=0.05,
                     horizontal_flip=True,
                     fill_mode='nearest')

# Train with data generator
# 以batch_size=2的速率无限生成增强后的数据
myGene = trainGenerator(1, 'data/tissue-train-pos0/', 'image', 'mask', data_gen_args)
print('\n\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n')
model = unet()
model_checkpoint = ModelCheckpoint(
    'unet_colonoscope.hdf5',
    monitor='loss',
    verbose=1,
    save_best_only=True)
print('------------------running the model.fit_generator() function----------------------')
model.fit_generator(
    myGene,
    steps_per_epoch=5,
    epochs=1,
    callbacks=[model_checkpoint])
print('\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
print('------------------finished the fit_generator() func--------------------------------')
print('=========================== generating test data ==================================')

# Train with npy file
#imgs_train,imgs_mask_train = geneTrainNpy("data/membrane/train/aug/","data/membrane/train/aug/")
#model.fit(imgs_train, imgs_mask_train, batch_size=2, nb_epoch=10, verbose=1,validation_split=0.2, shuffle=True, callbacks=[model_checkpoint])

# test your model and save predicted results
testGene = testGenerator("data/tissue-train-pos/")
results = model.predict_generator(testGene, steps=10, verbose=1)
# steps: 在停止之前，来自 generator 的总步数 (样本批次)
# predict_generator(
#     generator,
#     steps=None,
#     callbacks=None,
#     max_queue_size=10,
#     workers=1,
#     use_multiprocessing=False,
#     verbose=0
# )
# steps: Total number of steps (batches of samples) to yield from generator before stopping.
# max_queue_size: Maximum size for the generator queue.
saveResult("data/tissue-test-pos0/", results)
