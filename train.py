# Training a NestFuse network
# auto-encoder

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
import time
from tqdm import tqdm, trange
import scipy.io as scio
import random
import torch
from torch.optim import Adam
from torch.autograd import Variable
import utils
from net_repvgg import Repvgg_net
from args_fusion import args
import pytorch_msssim


EPSILON = 1e-5


def main():
	original_imgs_path = utils.list_images(args.dataset_ir)
	train_num = 80000
	original_imgs_path = original_imgs_path[:train_num]
	random.shuffle(original_imgs_path)
	img_flag = args.img_flag
	alpha_list = [10000]
	w_all_list = [[6.0, 3.0]]

	for w_w in w_all_list:
		w1, w2 = w_w
		for alpha in alpha_list:
			train(original_imgs_path, img_flag, alpha, w1, w2)


def train(original_imgs_path, img_flag, alpha, w1, w2):

	batch_size = args.batch_size
	# load network model
	nc = 1
	input_nc = args.input_nc
	output_nc = args.output_nc
	BTSFusion_model = Repvgg_net(input_nc, output_nc)

	if args.resume is not None:
		print('Resuming, initializing using weight from {}.'.format(args.resume))
		BTSFusion_model.load_state_dict(torch.load(args.resume))
		
	print(BTSFusion_model)
	optimizer = Adam(BTSFusion_model.parameters(), args.lr)
	mse_loss = torch.nn.MSELoss()
	ssim_loss = pytorch_msssim.msssim

	if args.cuda:
		BTSFusion_model.cuda()

	tbar = trange(args.epochs)
	print('Start training.....')

	# creating save path
	temp_path_model = os.path.join(args.save_model_dir)
	temp_path_loss  = os.path.join(args.save_loss_dir)
	if os.path.exists(temp_path_model) is False:
		os.mkdir(temp_path_model)

	if os.path.exists(temp_path_loss) is False:
		os.mkdir(temp_path_loss)

	temp_path_model_w = os.path.join(args.save_model_dir, str(w1))
	temp_path_loss_w  = os.path.join(args.save_loss_dir, str(w1))
	if os.path.exists(temp_path_model_w) is False:
		os.mkdir(temp_path_model_w)

	if os.path.exists(temp_path_loss_w) is False:
		os.mkdir(temp_path_loss_w)

	Loss_feature = []
	Loss_ssim = []
	Loss_all = []
	count_loss = 0
	all_ssim_loss = 0.
	all_fea_loss = 0.
	for e in tbar:
		print('Epoch %d.....' % e)
		# load training database
		image_set_ir, batches = utils.load_dataset(original_imgs_path, batch_size)
		BTSFusion_model.train()
		BTSFusion_model.cuda()
        
        
		count = 0
		for batch in range(batches):
			image_paths_ir = image_set_ir[batch * batch_size:(batch * batch_size + batch_size)]
			img_ir = utils.get_train_images_auto(image_paths_ir, height=args.HEIGHT, width=args.WIDTH, mode=img_flag)

			image_paths_vi = [x.replace('lwir', 'visible') for x in image_paths_ir]
			img_vi = utils.get_train_images_auto(image_paths_vi, height=args.HEIGHT, width=args.WIDTH, mode=img_flag)

			image_paths_latlrr = [x.replace('lwir', 'latlrr') for x in image_paths_ir]
			image_paths_latlrr = [x.replace('\\I', '\\saliency') for x in image_paths_latlrr]
			image_paths_latlrr = [x.replace('.jpg', '.png') for x in image_paths_latlrr]
			img_latlrr = utils.get_train_images_auto(image_paths_latlrr, height=args.HEIGHT, width=args.WIDTH, mode=img_flag)

			count += 1
			optimizer.zero_grad()
			img_ir = Variable(img_ir, requires_grad=False)
			img_vi = Variable(img_vi, requires_grad=False)
			img_latlrr = Variable(img_latlrr, requires_grad=False)

			if args.cuda:
				img_ir = img_ir.cuda()
				img_vi = img_vi.cuda()
				img_latlrr = img_latlrr.cuda()
			# encoder
			en_ir = BTSFusion_model.encoder_ir(img_ir)
			en_vi = BTSFusion_model.encoder(img_vi)
			f = BTSFusion_model.fusion(en_ir, en_vi)
			# decode
			outputs = BTSFusion_model.decoder(f)

			x_ir = Variable(img_ir.data.clone(), requires_grad=False)
			x_vi = Variable(img_vi.data.clone(), requires_grad=False)
			x_latlrr = Variable(img_latlrr.data.clone(), requires_grad=False)

			######################### LOSS FUNCTION #########################
			loss1_value = 0.
			loss2_value = 0.
			for output in outputs:

				fusion_detail,fusion_smoothed = utils.gf_loss(x_ir,x_vi)
				output_detail,output_smoothed = utils.gf_out(output)
				ssim_loss_temp2 = ssim_loss(output_detail, fusion_detail, normalize=True)
				max_input_pixel,mask = utils.PixelIntensityDecision(x_latlrr, x_ir, x_vi)
				pixel_loss_temp = ( mse_loss(output * mask, max_input_pixel * mask) * 4  + mse_loss((output - output * mask), (max_input_pixel - max_input_pixel * mask)) * 6 )/10
				loss1_value += alpha * (1 - ssim_loss_temp2)
				loss2_value += pixel_loss_temp

			loss1_value /= len(outputs)
			loss2_value /= len(outputs)

			# total loss
			total_loss = loss1_value + loss2_value
			total_loss.backward()
			optimizer.step()

			all_fea_loss += loss2_value.item() # 
			all_ssim_loss += loss1_value.item() # 
			if (batch + 1) % args.log_interval == 0:
				mesg = "{}\t Alpha: {} \tW-IR: {}\tEpoch {}:\t[{}/{}]\t ssim loss: {:.6f}\t fea loss: {:.6f}\t total: {:.6f}".format(
					time.ctime(), alpha, w1, e + 1, count, batches,
								  all_ssim_loss / args.log_interval,
								  all_fea_loss / args.log_interval,
								  (all_fea_loss + all_ssim_loss) / args.log_interval
				)
				tbar.set_description(mesg)
				Loss_ssim.append( all_ssim_loss / args.log_interval)
				Loss_feature.append(all_fea_loss / args.log_interval)
				Loss_all.append((all_fea_loss + all_ssim_loss) / args.log_interval)
				count_loss = count_loss + 1
				all_ssim_loss = 0.
				all_fea_loss = 0.

			if (batch + 1) % (200 * args.log_interval) == 0:
				# save model
				BTSFusion_model.eval()
				BTSFusion_model.cpu()
                
				save_model_filename = "Epoch_" + str(e) + "_iters_" + str(count) + "_alpha_" + str(alpha) + "_wir_" + str(w1) + "_wvi_" + str(w2) + ".model"
				save_model_path = os.path.join(args.save_model_dir, save_model_filename)
				torch.save(BTSFusion_model.state_dict(), save_model_path)
                
				# save loss data
				# pixel loss
				loss_data_ssim = Loss_ssim
				loss_filename_path = temp_path_loss_w + "/loss_ssim_epoch_" + str(args.epochs) + "_iters_" + str(count) + "_alpha_" + str(alpha) + "_wir_" + str(w1) + "_wvi_" + str(w2) + ".mat"
				scio.savemat(loss_filename_path, {'loss_ssim': loss_data_ssim})
				# SSIM loss
				loss_data_fea = Loss_feature
				loss_filename_path = temp_path_loss_w + "/loss_fea_epoch_" + str(args.epochs) + "_iters_" + str(count) + "_alpha_" + str(alpha) + "_wir_" + str(w1) + "_wvi_" + str(w2) + ".mat"
				scio.savemat(loss_filename_path, {'loss_fea': loss_data_fea})
				# all loss
				loss_data = Loss_all
				loss_filename_path = temp_path_loss_w + "/loss_all_epoch_" + str(args.epochs) + "_iters_" + str(count) + "_alpha_" + str(alpha) + "_wir_" + str(w1) + "_wvi_" + str(w2) + ".mat"
				scio.savemat(loss_filename_path, {'loss_all': loss_data})

				BTSFusion_model.train()
				BTSFusion_model.cuda()
                				
				tbar.set_description("\nCheckpoint, trained model saved at", save_model_path)

		# ssim loss
		loss_data_ssim = Loss_ssim
		loss_filename_path = temp_path_loss_w + "/Final_loss_ssim_epoch_" + str(
			args.epochs) + "_alpha_" + str(alpha) + "_wir_" + str(w1) + "_wvi_" + str(w2) + ".mat"
		scio.savemat(loss_filename_path, {'final_loss_ssim': loss_data_ssim})
		loss_data_fea = Loss_feature
		loss_filename_path = temp_path_loss_w + "/Final_loss_2_epoch_" + str(
			args.epochs) + "_alpha_" + str(alpha) + "_wir_" + str(w1) + "_wvi_" + str(w2) + ".mat"
		scio.savemat(loss_filename_path, {'final_loss_fea': loss_data_fea})
		# SSIM loss
		loss_data = Loss_all
		loss_filename_path = temp_path_loss_w + "/Final_loss_all_epoch_" + str(
			args.epochs) + "_alpha_" + str(alpha) + "_wir_" + str(w1) + "_wvi_" + str(w2) + ".mat"
		scio.savemat(loss_filename_path, {'final_loss_all': loss_data})
		# save model
		BTSFusion_model.eval()
		BTSFusion_model.cpu()
        
		save_model_filename = "Final_epoch_" + str(args.epochs) + "_alpha_" + str(alpha) + "_wir_" + str(
			w1) + "_wvi_" + str(w2) + ".model"
		save_model_path = os.path.join(args.save_model_dir, save_model_filename)
		torch.save(BTSFusion_model.state_dict(), save_model_path)
        
		print("\nDone, trained model saved at", save_model_path)


def check_paths(args):
	try:
		if not os.path.exists(args.vgg_model_dir):
			os.makedirs(args.vgg_model_dir)
		if not os.path.exists(args.save_model_dir):
			os.makedirs(args.save_model_dir)
	except OSError as e:
		print(e)
		sys.exit(1)



if __name__ == "__main__":
	main()
