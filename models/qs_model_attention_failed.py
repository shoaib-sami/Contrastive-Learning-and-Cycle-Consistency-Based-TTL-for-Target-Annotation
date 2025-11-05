import numpy as np
import torch
from .base_model import BaseModel
from . import networks_global, networks_local, networks_local_global,networks_global1
from .monce import MoNCELoss as PatchNCELoss
#from .patchnce import PatchNCELoss
import util.util as util



import itertools
import my_resnet 
from my_resnet import model
import my_resnet 
from my_resnetA import model1
import torch.nn as nn

class QSModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--QS_mode', type=str, default="global", choices='(global, local, local_global)')
        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--nce_idt', type=util.str2bool, default=True, help='use NCE loss for identity mapping: NCE(G(Y), Y))')
        parser.add_argument('--nce_layers', type=str, default='0,4,8,12,16', help='compute NCE loss on which layers')
        parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'], help='how to downsample the feature map')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")

        parser.set_defaults(pool_size=0)  # no image pooling

        opt, _ = parser.parse_known_args()

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>





        self.loss_names = ['G_A','G_B', 'D_real', 'D_fake', 'G', 'NCE']
        self.visual_names = ['real_A', 'fake_B','idt_A1' ,'rec_A', 'real_B', 'fake_A']
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]
        self.nce_layersB = [int(i) for i in self.opt.nce_layers.split(',')]
        self.iteration = 0


        if opt.nce_idt and self.isTrain:
            self.loss_names += ['NCE_Y']
            self.loss_names += ['cycle_A','cycle_B','NCE2','NCE3','D','idt_A','idt_B','D1','clf1','clf2'] #'cycle_A','cycle_B',
            self.visual_names += ['idt_B1','rec_B']

        if self.isTrain:
            self.model_names = ['G', 'F', 'D' ,'GB','FB'] #'FCA','FCB','FIA','FIB'
        else:  # during test time, only load G
            self.model_names = ['G']



        if self.opt.QS_mode == 'global':
            networks = networks_global
            print("Attention")
            networks1 = networks_global1
            

        elif self.opt.QS_mode == 'local':
            networks = networks_local
        else:
            networks = networks_local_global

        # define networks (both generator and discriminator)

        # define networks (both generator and discriminator)
        self.netG = networks1.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'our', opt.normG,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids) 
                        #networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)
        self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
        self.netGB = networks1.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'our', opt.normG,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids) 
        #networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)
        self.netFB = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
        #self.netFCA = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
        
        #self.netFCB = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
        
        #self.netFIA = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
        
        #self.netFIB = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)


        """ 

        PATH2 = '/media/shoaibmerajsami/SMS/ATR_database_october_2021/cycle_gan_july_2022_final/query-MoNCE/checkpoints/QS_M_V_L1_cycle_v5/latest_net_G.pth'  #'/media/shoaibmerajsami/SMS/ATR_database_october_2021/cycle_gan_july_2022_final/Modified_CUT_sep12/checkpoints/grumpycat_CUT_sep_17_cycle_V1/4_net_G (copy).pth'
        #PATH2 = '/media/shoaibmerajsami/SMS/cycle_gan_july_2022_final/august 25th 2022/pytorch-CycleGAN-and-pix2pix-master/models/summer2winter_yosemite.pth'
        print('...................................................................................')
        
        PATHB = '/media/shoaibmerajsami/SMS/ATR_database_october_2021/cycle_gan_july_2022_final/query-MoNCE/checkpoints/QS_M_V_L1_cycle_v5/latest_net_GB.pth'  #'/media/shoaibmerajsami/SMS/ATR_database_october_2021/cycle_gan_july_2022_final/Modified_CUT_sep12/checkpoints/grumpycat_CUT_sep_17_cycle_V1/4_net_GB (copy).pth'
        print(PATHB)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        state_dict = torch.load(PATH2)
        for k, v in state_dict.items():
            name = k#[7:] # remove 'module.' of dataparallel
            new_state_dict[name]=v
        #state_dict1 = [key[7:],value for key, value in weight1.items()]
        self.netG.load_state_dict(new_state_dict)





        new_state_dict1 = OrderedDict()
        state_dict1 = torch.load(PATHB)
        for k1, v1 in state_dict1.items():
            name1 = k1#[7:] # remove 'module.' of dataparallel
            new_state_dict1[name1]=v1
        #state_dict1 = [key[7:],value for key, value in weight1.items()]
        self.netGB.load_state_dict(new_state_dict1)




        #self.netGB.load_state_dict(torch.load(PATHB),strict=False)
        
        #self.netGB.load_state_dict(torch.load(PATHB),strict=False)


        """
        """
        self.netFCA = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
        self.netFCB = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
        self.netFIA = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
        self.netFIB = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
        """
        #self.netFB.load_state_dict(torch.load(PATHFB),strict=False)



        #self.netF = self.netF.to('cpu')
        #self.netF = torch.nn.DataParallel(self.netF)
        #self.netF = self.netF.cuda()
        #self.netF.load_state(torch.load(PATHF))
        #self.netF.load_state_dict(torch.load(PATHF))
        self.criterion1 = nn.CrossEntropyLoss()
        self.criterion2 = nn.CrossEntropyLoss()
        self.criterion3 = nn.CrossEntropyLoss()
        self.criterion4 = nn.CrossEntropyLoss()
        # optimizer = optim.Adam(model.parameters(), lr=0.0002)
        #self.optimizer1 = torch.optim.Adam(self.model1.parameters(), lr=0.0005, betas=(0.9, 0.999))
        #self.scheduler1 = torch.optim.lr_scheduler.ExponentialLR(self.optimizer1, gamma=0.92)
        self.model1 = model.cuda()
        self.model1= nn.DataParallel(self.model1) # device_ids=[0,1]
        self.model1.load_state_dict(torch.load('/media/shoaibmerajsami/SMS/ATR_database_october_2021/cycle_gan_july_2022_final/Modified_CUT_sep12/models/MWIR_66_image_ResNet18_6_Epoch_Accuracy_99_62.pth')["model"])
        
        
        self.model2 = model1.cuda()
        self.model2= nn.DataParallel(self.model2) # device_ids=[0,1]
        self.model2.load_state_dict(torch.load('/media/shoaibmerajsami/SMS/ATR_database_october_2021/cycle_gan_july_2022_final/CUT_September_4_2022/contrastive-unpaired-translation-master/Visible_66_image_ResNet18_5_Epoch99_28.pth')["model"])
        for param in self.model2.parameters():
            param.requires_grad = False
        for param2 in self.model2.parameters():
            param2.requires_grad = False


        if self.isTrain:
            self.netD = networks1.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netDB = networks1.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, self.gpu_ids)
            print("Hello")

            #self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
            #self.netDB = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
            # define loss functions
            self.criterionGAN = networks1.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = []
            self.criterionMoNCE = []
            self.criterionNCEB = []
            #self.criterionNCECA = []
            #self.criterionNCECB = []
            #self.criterionNCEIA = []
            #self.criterionNCEIB = []
            """
            
            self.criterionMoNCE = []
            self.criterionNCECA = []
            self.criterionNCECB = []
            self.criterionNCEIA = []
            self.criterionNCEIB = []
            """
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()

            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))
                self.criterionNCEB.append(PatchNCELoss(opt).to(self.device))
                #self.criterionNCECA.append(PatchNCELoss(opt).to(self.device))
                #self.criterionNCECB.append(PatchNCELoss(opt).to(self.device))
                #self.criterionNCEIA.append(PatchNCELoss(opt).to(self.device))
                #self.criterionNCEIB.append(PatchNCELoss(opt).to(self.device))
            """
            for nce_layer in self.nce_layers:
                self.criterionMoNCE.append(MoNCELoss(opt).to(self.device))
                self.criterionMoNCEB.append(MoNCELoss(opt).to(self.device))
            """

            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG.parameters(), self.netGB.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD.parameters(), self.netDB.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def data_dependent_initialize(self):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        
        bs_per_gpu = self.real_A.size(0) // max(len(self.opt.gpu_ids), 1)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.forward()                     # compute fake images: G(A)
        if self.opt.isTrain:
            self.compute_D_loss().backward()                  # calculate gradients for D
            self.compute_G_loss().backward()                   # calculate graidents for G
            if self.opt.lambda_NCE > 0.0:
                self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_F)
                #self.optimizer_FB = torch.optim.Adam(self.netFB.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
                #self.optimizers.append(self.optimizer_FB)
                self.optimizer_FB = torch.optim.Adam(self.netFB.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
               
                """ 
                self.optimizer_FCA = torch.optim.Adam(self.netFCA.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_FCA)
                self.optimizers.append(self.optimizer_FCA)
                self.optimizer_FCB = torch.optim.Adam(self.netFCB.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_FCB)     
                self.optimizer_FIA = torch.optim.Adam(self.netFIA.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_FIA)
                self.optimizer_FIB = torch.optim.Adam(self.netFIB.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_FIB) 
                """



    def optimize_parameters(self):
        # forward
        self.forward()

        # update D
        if self.iteration % 5 == 1:
            self.set_requires_grad(self.netD, True)
            self.optimizer_D.zero_grad()
            self.loss_D = self.compute_D_loss()
            self.loss_DB = self.compute_DB_loss()
            self.loss_D.backward()
            self.loss_DB.backward()
            self.optimizer_D.step()
            #self.optimizer_DB.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.zero_grad()
            self.optimizer_FB.zero_grad()
            """
            self.optimizer_FCA.zero_grad()
            self.optimizer_FCB.zero_grad()
            self.optimizer_FIA.zero_grad()
            self.optimizer_FIB.zero_grad()
            """

        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()
        self.optimizer_F.step()
        self.optimizer_FB.step()
        """
        self.optimizer_FCA.step()
        self.optimizer_FCB.step()
        self.optimizer_FIA.step()
        self.optimizer_FIB.step()
        """
       

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']




        self.image_paths_A = input['A_paths' if AtoB else 'B_paths']
        
        newlist = [xa.split('_')[-5][0:2]  for xa in self.image_paths_A ]
        self.label_A = newlist
        threshold = 1
        for i in range(len(newlist)):
            if newlist[i] == '2S':
                self.label_A[i] =1 - threshold
            if newlist[i] == 'BM':
                self.label_A[i] =2 - threshold
            if newlist[i] == 'BR':
                self.label_A[i] =3 - threshold
            if (newlist[i] == 'BT'):
                self.label_A[i] =4 - threshold
            if newlist[i] == 'D2':
                self.label_A[i] =5 - threshold
            if newlist[i] == 'MT':
                self.label_A[i] =6 - threshold
            if newlist[i] == 'Pi':
                self.label_A[i] =7 - threshold
            if newlist[i] == 'Sp':
                self.label_A[i] =8 - threshold
            if newlist[i] == 'T7':
                self.label_A[i] =9 - threshold
            if newlist[i] == 'ZS':
                self.label_A[i] =10 - threshold
            if (newlist[i] == 'Tr') or (newlist[i] =='cr') or (newlist[i] =='re'):
                print("A")
                self.label_A[i] =11 - threshold




        self.image_paths_B = input['B_paths' if AtoB else 'A_paths']
        newlist_B = [xb.split('_')[-5][0:2]  for xb in self.image_paths_B ]
        #newlist_C = [x.split('_')[-4][-4:] for x in self.image_paths_B ]
        #print(newlist_B)
        #print(newlist_C)
        for i in range(len(newlist_B)):
            if newlist_B[i] == '2S':
                newlist_B[i] =1 - threshold
            if newlist_B[i] == 'BM':
                newlist_B[i] =2 - threshold
            if newlist_B[i] == 'BR':
                newlist_B[i] =3 - threshold
            if (newlist_B[i] == 'BT'):
                newlist_B[i] =4 - threshold
            if newlist_B[i] == 'D2':
                newlist_B[i] =5 - threshold
            if newlist_B[i] == 'MT':
                newlist_B[i] =6 - threshold
            if newlist_B[i] == 'Pi':
                newlist_B[i] =7 - threshold
            if newlist_B[i] == 'Sp':
                newlist_B[i] =8 - threshold
            if newlist_B[i] == 'T7':
                newlist_B[i] =9 - threshold
            if newlist_B[i] == 'ZS':
                newlist_B[i] =10 - threshold
            if (newlist_B[i] == 'Tr') or (newlist_B[i] =='cr') or (newlist_B[i] =='re'):
                print('Danger')
                newlist_B[i] =11 - threshold





        #filename = self.image_paths[1].split('_')[-5]
        #d = filename.split('_')[-5]
        #self.label_A = newlist

        self.label_B = newlist_B
        self.labela = torch.as_tensor(self.label_A)

        self.labela = self.labela.cuda()
        #print(self.label_B)
        self.labelb = torch.as_tensor(self.label_B)
        self.labelb = self.labelb.cuda()

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""


        #self.feat_k = self.netG(self.real_A, self.nce_layers, encode_only=True)


        




        self.real = torch.cat((self.real_A, self.real_B), dim=0) if self.opt.nce_idt and self.opt.isTrain else self.real_A
        if self.opt.flip_equivariance:
            self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
            if self.flipped_for_equivariance:
                self.real = torch.flip(self.real, [3])



        self.fake_B, self.o1_b, self.o2_b, self.o3_b, self.o4_b, self.o5_b, self.o6_b, self.o7_b, self.o8_b, self.o9_b, self.o10_b, \
        self.a1_b, self.a2_b, self.a3_b, self.a4_b, self.a5_b, self.a6_b, self.a7_b, self.a8_b, self.a9_b, self.a10_b, \
        self.i1_b, self.i2_b, self.i3_b, self.i4_b, self.i5_b, self.i6_b, self.i7_b, self.i8_b, self.i9_b = self.netG(self.real_A)  # G_A(A)

        self.rec_A, self.rec_A_o1, self.rec_A_o2, self.rec_A_o3, self.rec_A_o4, self.rec_A_o5, self.rec_A_o6, self.rec_A_o7, self.rec_A_o8, self.rec_A_o9, self.rec_A_o10, \
        self.rec_A_a1, self.rec_A_a2, self.rec_A_a3, self.rec_A_a4, self.rec_A_a5, self.rec_A_a6, self.rec_A_a7, self.rec_A_a8, self.rec_A_a9, self.rec_A_a10, \
        self.rec_A_i1, self.rec_A_i2, self.rec_A_i3, self.rec_A_i4, self.rec_A_i5, self.rec_A_i6, self.rec_A_i7, self.rec_A_i8, self.rec_A_i9 = self.netGB(self.fake_B)   # G_B(G_A(A))

        self.fake_A, self.o1_a, self.o2_a, self.o3_a, self.o4_a, self.o5_a, self.o6_a, self.o7_a, self.o8_a, self.o9_a, self.o10_a, \
        self.a1_a, self.a2_a, self.a3_a, self.a4_a, self.a5_a, self.a6_a, self.a7_a, self.a8_a, self.a9_a, self.a10_a, \
        self.i1_a, self.i2_a, self.i3_a, self.i4_a, self.i5_a, self.i6_a, self.i7_a, self.i8_a, self.i9_a = self.netGB(self.real_B)  # G_B(B)
        self.rec_B, _, _, _, _, _, _, _, _, _, _, \
        _, _, _, _, _, _, _, _, _, _, \
        _, _, _, _, _, _, _, _, _ = self.netG(self.fake_A)   # G_A(G_B(B))


        """



        self.fake = self.netG(self.real)
        self.fake_B = self.fake[:self.real_A.size(0)]

        self.idt_B = self.fake[self.real_A.size(0):]
        

        self.idt_A1 = self.netGB(self.real_A)

        self.idt_B1 = self.netG(self.real_B)

        self.rec_A = self.netGB(self.real_A)#(self.fake_B)   # G_B(G_A(A))
        self.fake_A = self.netGB(self.real_B)  # G_B(B)
        self.rec_B = self.netG(self.real_B)#(self.fake_A)
        self.feat_k = self.netG(self.real_A, self.nce_layers, encode_only=True)
        """

    def backward_D(self):
        fake = self.fake_B.detach()
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(fake)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        # Real
        self.pred_real = self.netD(self.real_B)
        loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_real = loss_D_real.mean()

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D


    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        fake = self.fake_B.detach()
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(fake)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        # Real
        self.pred_real = self.netD(self.real_B)
        loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_real = loss_D_real.mean()

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D

    def compute_DB_loss(self):
        """Calculate GAN loss for the discriminator"""
        fake1 = self.fake_A.detach()
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake1 = self.netDB(fake1)
        self.loss_D_fake1 = self.criterionGAN(pred_fake1, False).mean()
        # Real
        self.pred_real1 = self.netDB(self.real_A)
        loss_D_real1 = self.criterionGAN(self.pred_real1, True)
        self.loss_D_real1 = loss_D_real1.mean()

        # combine loss and calculate gradients
        self.loss_D1 = (self.loss_D_fake1 + self.loss_D_real1) * 0.5
        return self.loss_D1



    def compute_G_loss(self):




        lambda_A = 10
        lambda_B = 10
        lambda_idt = 0.5
        fake = self.fake_B
        data2_fake_B = self.fake_B
        
        data4 = self.fake_A
        #data2 = data2.view(-1, 66, 66)
        #data4 = data4.view(-1, 66, 66)
        
        
        data2_fake_B = data2_fake_B.cuda()        
        
        
        data4 = data4.cuda()
        
       
        self.scores2_fake_B = self.model2(data2_fake_B)
        self.scores4 = self.model1(data4)
        #self.classification_loss1 = self.criterion1(self.scores1,self.labela)*0.2
        self.classification_loss2 = self.criterion2(self.scores2_fake_B,self.labela)*4
        #self.classification_loss3 = self.criterion3(self.scores3,self.labelb)*0.2
        self.classification_loss4 = self.criterion4(self.scores4,self.labelb)*4

        # First, G(A) should fake the discriminator
        if self.opt.lambda_GAN > 0.0:
            pred_fake = self.netD(self.fake_B)
            self.loss_G_A = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN 
            self.loss_G_GAN = self.loss_G_A +  self.classification_loss2  + self.classification_loss4
            self.iteration +=1

        else:
            self.loss_G_GAN = 0.0

        if self.opt.lambda_NCE > 0.0:
            self.loss_NCE = self.calculate_NCE_loss(self.real_A, self.fake_B)
        else:
            self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0

        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
            self.loss_NCE_Y = self.calculate_NCE_loss(self.real_B, self.idt_B)
            loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y) * 0.5
        else:
            loss_NCE_both = self.loss_NCE

        lambda_idt =0.5


        #self.idt_Anew = self.netGB(self.real_B)
        self.loss_idt_A = self.criterionIdt(self.idt_A1, self.real_A) * lambda_B * lambda_idt
        # G_B should be identity if real_A is fed: ||G_B(A) - A||
        #self.idt_B = self.netGB(self.real_A)
        self.loss_idt_B = self.criterionIdt(self.idt_B1, self.real_B) *lambda_A * lambda_idt
    


        # GAN loss D_A(G_A(A))
        #self.loss_G_A = self.criterionGAN(self.netD(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netDB(self.fake_A), True).mean() * self.opt.lambda_GAN 

        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        #self.loss_cycle_CA = self.calculate_NCE_lossCB( self.rec_A, self.real_A) 
        #self.loss_cycle_CB = self.calculate_NCE_lossCA(self.rec_B, self.real_B) 
        #self.loss_NCE_both_cycle = (self.loss_cycle_CA + self.loss_cycle_CB) * 0.5

        self.loss_NCE2 = self.calculate_NCE_lossB(self.real_B, self.fake_A)
        self.loss_NCE3 = self.calculate_NCE_lossB(self.real_A, self.idt_A1)
        loss_NCE_both2 = (self.loss_NCE2 + self.loss_NCE3) * 0.5

        self.loss_G = self.loss_G_GAN + loss_NCE_both + loss_NCE_both2 + self.loss_G_B + self.loss_idt_A + self.loss_idt_B + self.loss_cycle_A + self.loss_cycle_B 
        
        self.loss_clf1 = self.classification_loss2
        self.loss_clf2 = self.classification_loss4
        if self.iteration % 5000 ==2:
            print("Classification Loss")
            print(self.classification_loss2 + self.classification_loss4)
            print("NCE loss GenA ")
            print(loss_NCE_both)
            print("NCE loss GenB")
            print(loss_NCE_both2)
            print("Cycle All")
            print(self.loss_cycle_A + self.loss_cycle_B )
            print("Generator loss A")
            print(self.loss_G_GAN)
            print("Generator Loss B")
            print(self.loss_G_B)
            print("IDT loss A")
            print(self.loss_idt_A)
            print("IDT Loss B")
            print(self.loss_idt_B)
            print("GAN Loss total")
            print(self.loss_G)

        return self.loss_G










    def calculate_NCE_loss(self, src, tgt):
        n_layers = len(self.nce_layers)
        feat_q = self.netG(tgt, self.nce_layers, encode_only=True)

        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        feat_k = self.netG(src, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids, attn_mats = self.netF(feat_k, self.opt.num_patches, None, None)
        feat_q_pool, _, _ = self.netF(feat_q, self.opt.num_patches, sample_ids, attn_mats)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers


    def calculate_NCE_lossB(self, src, tgt):
        n_layers = len(self.nce_layers)
        feat_q = self.netGB(tgt, self.nce_layers, encode_only=True)

        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        feat_k = self.netGB(src, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids, attn_mats = self.netFB(feat_k, self.opt.num_patches, None, None)
        feat_q_pool, _, _ = self.netFB(feat_q, self.opt.num_patches, sample_ids, attn_mats)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCEB, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers

    """

    def calculate_NCE_lossCA(self, src, tgt):
        n_layers = len(self.nce_layers)
        feat_q = self.netG(tgt, self.nce_layers, encode_only=True)

        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        feat_k = self.netG(src, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids, attn_mats = self.netFCA(feat_k, self.opt.num_patches, None, None)
        feat_q_pool, _, _ = self.netFCA(feat_q, self.opt.num_patches, sample_ids, attn_mats)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCECA, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers




    def calculate_NCE_lossCB(self, src, tgt):
        n_layers = len(self.nce_layers)
        feat_q = self.netGB(tgt, self.nce_layers, encode_only=True)

        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        feat_k = self.netGB(src, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids, attn_mats = self.netFCB(feat_k, self.opt.num_patches, None, None)
        feat_q_pool, _, _ = self.netFCB(feat_q, self.opt.num_patches, sample_ids, attn_mats)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCECB, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers







    def calculate_NCE_lossIA(self, src, tgt):
        n_layers = len(self.nce_layers)
        feat_q = self.netG(tgt, self.nce_layers, encode_only=True)

        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        feat_k = self.netG(src, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids, attn_mats = self.netFIA(feat_k, self.opt.num_patches, None, None)
        feat_q_pool, _, _ = self.netFIA(feat_q, self.opt.num_patches, sample_ids, attn_mats)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCEIA, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers




    def calculate_NCE_lossIB(self, src, tgt):
        n_layers = len(self.nce_layers)
        feat_q = self.netGB(tgt, self.nce_layers, encode_only=True)

        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        feat_k = self.netGB(src, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids, attn_mats = self.netFIB(feat_k, self.opt.num_patches, None, None)
        feat_q_pool, _, _ = self.netFIB(feat_q, self.opt.num_patches, sample_ids, attn_mats)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCEIB, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers

    """

