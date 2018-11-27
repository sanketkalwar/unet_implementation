import mxnet as mx
import numpy as np
smooth = 1.
size = 572
batch_size = 10
epoch =10
data_shape = (batch_size,1,size,size)


def Loss(label,output):
	#label = mx.sym.slice_like(label,output,name='crop_label')
	intersection = mx.sym.sum(mx.sym.broadcast_mul(label,output,name='mul'),name='intersection')
	total = mx.sym.sum(label)+mx.sym.sum(output)+smooth
	return -mx.sym.broadcast_div((2*intersection+smooth),total)

def U_net():
	data = mx.sym.Variable('data')
	label = mx.sym.Variable('label')
	
	conv1_1  = mx.sym.Convolution(data,num_filter=64,kernel=(3,3),name='conv1')
	batch1_1 = mx.sym.BatchNorm(conv1_1,name='batch1')
	relu1_1  = mx.sym.Activation(batch1_1,act_type='relu',name='relu1')
	conv1_2  = mx.sym.Convolution(relu1_1,num_filter=64,kernel=(3,3),name='conv2')
	batch1_2 = mx.sym.BatchNorm(conv1_2,name='batch2')
	relu1_2  = mx.sym.Activation(batch1_2,act_type='relu',name='relu2')
	pool1    = mx.sym.Pooling(relu1_2,kernel=(2,2),name='pool1',stride=(2,2))
	
	conv2_1  = mx.sym.Convolution(pool1,num_filter=128,kernel=(3,3),name='conv3')
	batch2_1 = mx.sym.BatchNorm(conv2_1,name='batch3')
	relu2_1  = mx.sym.Activation(batch2_1,act_type='relu',name='relu3')
	conv2_2  = mx.sym.Convolution(batch2_1,num_filter=128,kernel=(3,3),name='conv4')
	batch2_2 = mx.sym.BatchNorm(conv2_2,name='batch4')
	relu2_2  = mx.sym.Activation(batch2_2,act_type='relu',name='relu4')
	pool2    = mx.sym.Pooling(relu2_2,kernel=(2,2),name='pool2',stride=(2,2))
	
	conv3_1  = mx.sym.Convolution(pool2,num_filter=256,kernel=(3,3),name='conv5')
	batch3_1 = mx.sym.BatchNorm(conv3_1,name='batch5')
	relu3_1  = mx.sym.Activation(batch3_1,act_type='relu',name='relu5')
	conv3_2  = mx.sym.Convolution(relu3_1,num_filter=256,kernel=(3,3),name='conv6')
	batch3_2 = mx.sym.BatchNorm(conv3_2,name='batch6')
	relu3_2  = mx.sym.Activation(batch3_2,act_type='relu',name='relu6')
	pool3    = mx.sym.Pooling(relu3_2,kernel=(2,2),name='pool3',stride=(2,2))
	
	conv4_1  = mx.sym.Convolution(pool3,num_filter=512,kernel=(3,3),name='conv7')
	batch4_1 = mx.sym.BatchNorm(conv4_1,name='batch7')
	relu4_1  = mx.sym.Activation(batch4_1,act_type='relu',name='relu7')
	conv4_2  = mx.sym.Convolution(relu4_1,num_filter=512,kernel=(3,3),name='conv8')
	batch4_2 = mx.sym.BatchNorm(conv4_2,name='batch8')
	relu4_2  = mx.sym.Activation(batch4_2,act_type='relu',name='relu8')
	pool4    = mx.sym.Pooling(relu4_2,kernel=(2,2),name='pool4',stride=(2,2))
	
	conv5_1  = mx.sym.Convolution(pool4,num_filter=1024,kernel=(3,3),name='conv9')
	batch5_1 = mx.sym.BatchNorm(conv5_1,name='batch9')
	relu5_1  = mx.sym.Activation(batch5_1,act_type='relu',name='relu9')
	conv5_2  = mx.sym.Convolution(relu5_1,num_filter=1024,kernel=(3,3),name='conv10')
	batch5_2 = mx.sym.BatchNorm(conv5_2,name='batch10')
	relu5_2  = mx.sym.Activation(batch5_2,act_type='relu',name='relu10')
	
	#First Deconvolution
	upool5   = mx.sym.Deconvolution(relu5_2,num_filter=512,kernel=(2,2),name='upool5',stride=(2,2))
	crop5    = mx.sym.slice_like(relu4_2,upool5,name="crop5")
	concat5  = mx.sym.concat(*[upool5,crop5],dim=1,name='concat5')
	
	conv6_1  = mx.sym.Convolution(concat5,num_filter=512,kernel=(3,3),name='conv11')
	batch6_1 = mx.sym.BatchNorm(conv6_1,name='batch11') 
	relu6_1  = mx.sym.Activation(batch6_1,act_type='relu',name='relu11')
	conv6_2  = mx.sym.Convolution(relu6_1,num_filter=512,kernel=(3,3),name='conv12')
	batch6_2 = mx.sym.BatchNorm(conv6_2,name='batch12')
	relu6_2  = mx.sym.Activation(batch6_2,act_type='relu',name='relu12')
	
	#Second Deconvolution
	upool6   = mx.sym.Deconvolution(relu6_2,num_filter=256,kernel=(2,2),name='upool6',stride=(2,2))
	crop6    = mx.sym.slice_like(relu3_2,upool6,name='crop6')
	concat6  = mx.sym.concat(*[upool6,crop6],dim=1,name='concat6')
	
	conv7_1  = mx.sym.Convolution(concat6,num_filter=256,kernel=(3,3),name='conv13')
	batch7_1 = mx.sym.BatchNorm(conv7_1,name='batch13')
	relu7_1  = mx.sym.Activation(batch7_1,act_type='relu',name='relu13')
	conv7_2  = mx.sym.Convolution(relu7_1,num_filter=256,kernel=(3,3),name='conv14')
	batch7_2 = mx.sym.BatchNorm(conv7_2,name='batch14')
	relu7_2  = mx.sym.Activation(batch7_2,act_type='relu',name='relu14')

	#Third Deconvolution
	upool7   = mx.sym.Deconvolution(relu7_2,num_filter=128,kernel=(2,2),name='upool7',stride=(2,2))
	crop7 = mx.sym.slice_like(relu2_2,upool7,name='crop7')
	concat7  = mx.sym.concat(*[upool7,crop7],dim=1,name='concat7')
	
	conv8_1  = mx.sym.Convolution(concat7,num_filter=128,kernel=(3,3),name='conv15')
	batch8_1 = mx.sym.BatchNorm(conv8_1,name='batch15')
	relu8_1  = mx.sym.Activation(batch8_1,act_type='relu',name='relu15')
	conv8_2  = mx.sym.Convolution(relu8_1,num_filter=128,kernel=(3,3),name='conv16')
	batch8_2 = mx.sym.BatchNorm(conv8_2,name='batch16')
	relu8_2  = mx.sym.Activation(batch8_2,act_type='relu',name='relu16')

	#Fourth Deconvolution
	upool8   = mx.sym.Deconvolution(relu8_2,num_filter=64,kernel=(2,2),name='upool8',stride=(2,2))
	crop8    = mx.sym.slice_like(relu1_2,upool8,name='crop8')
	concat8  = mx.sym.concat(*[upool8,crop8],dim=1,name='concat8')
	
	conv9_1  = mx.sym.Convolution(concat8,num_filter=64,kernel=(3,3),name='conv17')
	batch9_1 = mx.sym.BatchNorm(conv9_1,name='batch17')
	relu9_1  = mx.sym.Activation(batch9_1,act_type='relu',name='relu17')
	conv9_2  = mx.sym.Convolution(relu9_1,num_filter=64,kernel=(3,3),name='conv18')
	batch9_2 = mx.sym.BatchNorm(conv9_2,name='batch18')
	relu9_2  = mx.sym.Activation(batch9_2,act_type='relu',name='relu18')

	conv10_1 = mx.sym.Convolution(relu9_2,num_filter=1,kernel=(1,1),name='conv20')
	batch10_1 = mx.sym.BatchNorm(conv10_1,name='batch20')
	out_flat = mx.sym.Flatten(batch10_1,name='out_flat')
	softmax  = mx.sym.softmax(out_flat,name='softmax')
	print(mx.sym.size_array(softmax))
	loss= mx.sym.MakeLoss(Loss(label,out_flat),normalization='batch')
	mask_output = mx.sym.BlockGrad(softmax,name='mask_output')
	output = mx.sym.Group([loss,mask_output])
	return output



if __name__ == '__main__':
	u_net = U_net()
	print(u_net)
	#mx.viz.p
	data = np.load('X_train.npy')
	data_mean= np.mean(data)
	data_std = np.std(data)
	data = data- data_mean
	data =data/data_std
	
	label = np.load('Y_train.npy')
	
	#print(label_std)
	label = label/255
	print("label size:",label.shape)
	label= label[:,:,:388,:388]
	label = label.reshape((label.shape[0],label.shape[2]*label.shape[3]))
	print("label",label.shape)
	net = mx.mod.Module(u_net,context=mx.gpu(),data_names=('data',),label_names=('label',))
	train_iter = mx.io.NDArrayIter(data[:400,:,:],label[:400,:],batch_size=batch_size,shuffle=True)
	val_iter   = mx.io.NDArrayIter(data[400:,:,:],label[400:,:],batch_size=batch_size,shuffle=True)
	net.bind(data_shapes=[['data',train_iter.data[0][1].asnumpy().shape]],label_shapes=[['label',train_iter.label[0][1].asnumpy().shape]])
	print('hii')
	print("train x:",train_iter.label[0][1].asnumpy().shape)
	net.init_params(initializer=mx.init.Xavier())
	#using adam optimizer
	net.init_optimizer(optimizer='adam',optimizer_params=(('learning_rate',0.001),('beta1',0.9),('beta2',0.999),('epsilon',1e-08)))
	metric = mx.metric.create('acc')
	temp_loss=[]
	for i in range(epoch):
		train_iter.reset()
		metric.reset()
		for batch in train_iter:
			net.forward(batch,is_train=True)
			#print(np.where(batch.label[0].asnumpy()==1))
			#net.update_metric(metric,batch.label[0])
			#print("loss",net.get_outputs())
			#np.save("tt.npy",tt)
			
			#temp_loss.append(tt)
			net.backward()
			net.update()
		#actual_loss=mx.ndarray.mean(loss)
		print("Epoch",i)
	print("out")

