import os, torch, argparse, math
import numpy as np
from copy import deepcopy
from tqdm import tqdm, trange

from vq import VectorQuantize
from utils import read_ply_data, write_ply_data, load_vqgaussian


def parse_args():
    parser = argparse.ArgumentParser(description="vectree quantization")
    parser.add_argument("--important_score_npz_path", type=str, default='room')
    parser.add_argument("--input_path", type=str, default='room/iteration_40000/point_cloud.ply')
    
    parser.add_argument("--save_path", type=str, default='./output/room')  
    parser.add_argument("--no_load_data", type=bool, default=False)
    parser.add_argument("--no_save_ply", type=bool, default=False)
    parser.add_argument("--sh_degree", type=int, default=2)

    parser.add_argument("--iteration_num", type=float, default=1000)
    parser.add_argument("--vq_ratio", type=float, default=0.6)
    parser.add_argument("--codebook_size", type=int, default=2**13)  # 2**13 = 8192
    parser.add_argument("--no_IS", type=bool, default=False)
    parser.add_argument("--vq_way", type=str, default='half') 
    opt = parser.parse_args() 
    return opt
    

class Quantization():
    def __init__(self, opt):
        """
        VecTree/VQ 压缩入口。

        输入 point_cloud.ply 会先被 read_ply_data 展开成二维属性矩阵，列大致为：
            xyz(3), normal(3), f_dc(3), f_rest(...), opacity(1), scale(3), rot(4)

        本类只对 SH 外观特征做向量量化：
            self.feats_bak: 完整属性，后续保存 xyz/opacity/scale/rot 时还要用；
            self.feats:     f_dc + f_rest，即需要被 VQ 的颜色/SH 部分。

        量化策略是 importance-aware：
            高重要性 Gaussian 的 SH 直接保存为 half；
            低重要性 Gaussian 的 SH 用 codebook index 表示。
        """
        
        # ----- load ply data -----
        if opt.sh_degree == 3:
            self.sh_dim = 3+45
        elif opt.sh_degree == 2:
            self.sh_dim = 3+24

        self.feats = read_ply_data(opt.input_path)
        self.feats = torch.tensor(self.feats)
        self.feats_bak = self.feats.clone()
        # 跳过 xyz(3)+normal(3)，只取 SH 相关特征：
        #   degree=2: f_dc(3)+f_rest(24)
        #   degree=3: f_dc(3)+f_rest(45)
        self.feats = self.feats[:, 6:6+self.sh_dim]

        # ----- define model -----
        # VectorQuantize 内部维护一个 EMA 更新的 codebook。每个低重要性 Gaussian
        # 的 SH 向量会被替换为最近 codebook 向量，并只保存 index。
        self.model_vq = VectorQuantize(
                    dim = self.feats.shape[1],              
                    codebook_size = opt.codebook_size,
                    decay = 0.8,                            
                    commitment_weight = 1.0,                
                    use_cosine_sim = False,
                    threshold_ema_dead_code=0,
                ).to(device)
        
        # ----- other -----
        self.save_path = opt.save_path
        self.ply_path = opt.save_path
        self.imp_path = opt.important_score_npz_path
        self.high = None
        self.VQ_CHUNK = 80000
        self.k_expire = 10        
        self.vq_ratio = opt.vq_ratio

        self.no_IS = opt.no_IS
        self.no_load_data = opt.no_load_data
        self.no_save_ply = opt.no_save_ply
   
        self.codebook_size = opt.codebook_size
        self.iteration_num = opt.iteration_num
        self.vq_way = opt.vq_way

        # ----- print info -----
        print("\n================== Print Info ================== ")
        print("Input_feats_shape: ", self.feats_bak.shape)
        print("VQ_feats_shape: ", self.feats.shape)
        print("SH_degree: ", opt.sh_degree)
        print("Quantization_ratio: ", opt.vq_ratio)
        print("Add_important_score: ", opt.no_IS==False)
        print("Codebook_size: ", opt.codebook_size)
        print("================================================ ")

    @torch.no_grad()
    def calc_vector_quantized_feature(self):
        """
        apply vector quantize on gaussian attributes and return vq indexes

        对所有 SH 特征应用训练好的 codebook，返回量化后的特征和 codebook index。

        这里按 CHUNK 分块是为了避免一次把全部 Gaussian 送进 VQ 造成显存峰值过高。
        """
        CHUNK = 8192
        feat_list = []
        indice_list = []
        self.model_vq.eval()
        self.model_vq._codebook.embed.half().float()   #
        for i in tqdm(range(0, self.feats.shape[0], CHUNK)):
            feat, indices, commit = self.model_vq(self.feats[i:i+CHUNK,:].unsqueeze(0).to(device))
            indice_list.append(indices[0])
            feat_list.append(feat[0])
        self.model_vq.train()
        all_feat = torch.cat(feat_list).half().float()  # [num_elements, feats_dim]
        all_indice = torch.cat(indice_list)             # [num_elements, 1]
        return all_feat, all_indice


    @torch.no_grad()
    def fully_vq_reformat(self):  
        """
        把量化结果重排并写成 extreme_saving 目录。

        保存格式拆成几类文件：
            metadata.npz:      点数、原始维度、codebook 维度等元信息；
            codebook.npz:      VQ codebook；
            vq_indexs.npz:     低重要性 Gaussian 对应的 codebook index，比特打包；
            non_vq_mask.npz:   哪些 Gaussian 不做 VQ；
            non_vq_feats.npz:  高重要性 Gaussian 的原始 SH；
            other_attribute.npz: opacity + scale + rotation；
            xyz.npz:           Gaussian 中心坐标。

        渲染量化模型时，load_vqgaussian() 会把这些文件重新拼回完整属性矩阵。
        """

        print("\n=============== Start vector quantize ===============")
        all_feat, all_indice = self.calc_vector_quantized_feature()

        if self.save_path is not None:
            save_path = self.save_path
            os.makedirs(f'{save_path}/extreme_saving', exist_ok=True)

            # ----- save basic info -----
            metadata = dict()
            metadata['input_pc_num'] = self.feats_bak.shape[0]  
            metadata['input_pc_dim'] = self.feats_bak.shape[1]  
            metadata['codebook_size'] = self.codebook_size
            metadata['codebook_dim'] = self.sh_dim
            np.savez_compressed(f'{save_path}/extreme_saving/metadata.npz', metadata=metadata)

            # ===================================================== save vq_SH =============================================
            # ----- save mapping_index (vq_index) -----
            def dec2bin(x, bits):
                mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, x.dtype)
                return x.unsqueeze(-1).bitwise_and(mask).ne(0).float()    
            # vq indice was saved in according to the bit length
            # 只保存进入 VQ 的 Gaussian 的 index。codebook_size=2^k 时，每个 index 可用 k bit 表示；
            # np.packbits 会进一步按 bit 打包，减少磁盘大小。
            self.codebook_vq_index = all_indice[torch.logical_xor(self.all_one_mask,self.non_vq_mask)]                             # vq_index
            bin_indices = dec2bin(self.codebook_vq_index, int(math.log2(self.codebook_size))).bool().cpu().numpy()                 # mapping_index
            np.savez_compressed(f'{save_path}/extreme_saving/vq_indexs.npz',np.packbits(bin_indices.reshape(-1)))               
            
            # ----- save codebook -----                                           
            codebook = self.model_vq._codebook.embed.cpu().half().numpy().squeeze(0)                                                 
            np.savez_compressed(f'{save_path}/extreme_saving/codebook.npz', codebook)

            # ----- save keep mask (non_vq_feats_index)-----
            np.savez_compressed(f'{save_path}/extreme_saving/non_vq_mask.npz',np.packbits(self.non_vq_mask.reshape(-1).cpu().numpy()))

            # ===================================================== save non_vq_SH =============================================
            # 高重要性 Gaussian 不经过 codebook，直接保存 SH 特征，通常转 half 降低大小。
            # 这部分是质量优先的保留集。
            non_vq_feats = self.feats_bak[self.non_vq_mask, 6:6+self.sh_dim]       
            wage_non_vq_feats = self.wage_vq(non_vq_feats)
            np.savez_compressed(f'{save_path}/extreme_saving/non_vq_feats.npz', wage_non_vq_feats) 

            # =========================================== save xyz & other attr(opacity + 3*scale + 4*rot) ====================================
            # 几何和 alpha 不做 VQ，否则很容易引入结构性伪影；这里只做 half/原精度保存。
            other_attribute = self.feats_bak[:, -8:]
            wage_other_attribute = self.wage_vq(other_attribute)
            np.savez_compressed(f'{save_path}/extreme_saving/other_attribute.npz', wage_other_attribute)

            xyz = self.feats_bak[:, 0:3]
            np.savez_compressed(f'{save_path}/extreme_saving/xyz.npz', xyz)  
            

        # zip everything together to get final size
        os.system(f"zip -r {save_path}/extreme_saving.zip {save_path}/extreme_saving")
        size = os.path.getsize(f'{save_path}/extreme_saving.zip')
        size_MB = size / 1024.0 / 1024.0
        print("Size = {:.2f} MB".format(size_MB))
            
        return all_feat, all_indice
    
    def load_f(self, path, name, allow_pickle=False,array_name='arr_0'):
        return np.load(os.path.join(path, name),allow_pickle=allow_pickle)[array_name]

    def wage_vq(self, feats):
        # 默认用 half 精度保存非 codebook 数据，是质量和文件大小之间的折中。
        if self.vq_way == 'half':        
            return feats.half()
        else:
            return feats
    
    def quantize(self):
        if self.no_IS:                                                      #  no important score
            # 没有重要性分数时退化为均匀对待所有 Gaussian。
            importance = np.ones((self.feats.shape[0]))                     
        else:
            # imp_score.npz 通常来自 prune_finetune.py 或 distill_train.py 的最后一次统计。
            importance = self.load_f(self.imp_path, 'imp_score.npz')

        ###################################################
        only_vq_some_vector = True
        if only_vq_some_vector:
            tensor_importance = torch.tensor(importance)
            # vq_ratio 表示进入 VQ 的比例；1-vq_ratio 是直接保留的高重要性比例。
            # non_vq_mask=True 的点会保存原始 SH，不参与 codebook 近似。
            large_val, large_index = torch.topk(tensor_importance, k=int(tensor_importance.shape[0] * (1-self.vq_ratio)), largest=True) 
            self.all_one_mask = torch.ones_like(tensor_importance).bool()     
            self.non_vq_mask = torch.zeros_like(tensor_importance).bool()         
            self.non_vq_mask[large_index] = True                         
        self.non_vq_index = large_index

        IS_non_vq_point = large_val.sum()
        IS_all_point = tensor_importance.sum()
        IS_percent = IS_non_vq_point/IS_all_point
        print("IS_percent: ", IS_percent)

        #=================== Codebook initialization & Update codebook ====================
        self.model_vq.train()
        with torch.no_grad():
            # vq_mask=True 表示低重要性、需要用 codebook 近似的 Gaussian。
            self.vq_mask = torch.logical_xor(self.all_one_mask, self.non_vq_mask)                  
            feats_needs_vq = self.feats[self.vq_mask].clone()                                       
            imp = tensor_importance[self.vq_mask].float()                                        
            k = self.k_expire                                                              
            if k > self.model_vq.codebook_size:
                k = 0            
            for i in trange(self.iteration_num):
                # 随机采样一个大 batch 更新 codebook。weight 使用重要性分数，
                # 使高重要性的 VQ 样本对 codebook EMA 更新影响更大。
                indexes = torch.randint(low=0, high=feats_needs_vq.shape[0], size=[self.VQ_CHUNK])         
                vq_weight = imp[indexes].to(device)
                vq_feature = feats_needs_vq[indexes,:].to(device)
                quantize, embed, loss = self.model_vq(vq_feature.unsqueeze(0), weight=vq_weight.reshape(1,-1,1))

                # 手动“复活”使用率最低的 k 个 code：用当前 batch 中最重要的样本替换它们。
                # 这能减少死码本条目，提高 codebook 对关键外观的覆盖。
                replace_val, replace_index = torch.topk(self.model_vq._codebook.cluster_size, k=k, largest=False)      
                _, most_important_index = torch.topk(vq_weight, k=k, largest=True)
                self.model_vq._codebook.embed[:,replace_index,:] = vq_feature[most_important_index,:]

        #=================== Apply vector quantization ====================
        all_feat, all_indices = self.fully_vq_reformat()

    def dequantize(self):
        # 量化完成后立即反量化写回 point_cloud.ply，方便用普通 render.py 验证质量。
        print("\n==================== Load saved data & Dequantize ==================== ")
        dequantized_feats = load_vqgaussian(os.path.join(self.save_path,'extreme_saving'), device=device)

        if self.no_save_ply == False:
            os.makedirs(f'{self.ply_path}/', exist_ok=True)
            write_ply_data(dequantized_feats.cpu().numpy(), self.ply_path, self.sh_dim)


if __name__=='__main__':
    opt = parse_args()
    device = torch.device('cuda')
    vq = Quantization(opt)

    vq.quantize()
    vq.dequantize()
    
    print("All done!")

