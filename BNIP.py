import sys
sys.path.append(r'abspath/Beauty_Algorithm')
from Beauty_Algorithm_v2 import image_processing
from edit_cli import *
#from NSGA_II import *
from FaceNet import facenet
import clip
from CLIP import clip_model
from CLIP import clip_img_score

from loguru import logger
import torch
import cv2
import os
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt

import torch
import random
import numpy as np

# 计算ssim值
def calculate_ssim(img1, img2):
    # 将图像转换为灰度图
    # gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 计算SSIM指标
    # PSNR越大表示图像质量越好，SSIM越大表示两图像越相似
    # MSE的值越小，说明两幅图像越相似。
    ssim_score = ssim(img1, img2, channel_axis=2)#如果是彩色图像，channel_axis=2.如果是灰度图像，channel_axis就不需要设置。
    psnr_score = psnr(img1, img2)
    return ssim_score

# 20240412更改适应度函数
def fitness_function(img,oriImage_path,ori_imageEme,oriEditImage,oriEditFeature,population,generation,save_dir):
    
    logger.info(f"******************开始第{generation}代种群！！！****************************")
    # 保存图片的所有相似度量值
    o_Distance = []
    o_Value = []
    o_CValue = []
    e_Distance = []
    e_Value = []
    e_DValue = []
    # 保存图片的适应度值
    fitness_values1 = []
    fitness_values2 = []
    
    ori_image,pro_images,proImage_paths = image_processing(img,population,generation)
    # # 通过facenet获得图像的特征向量。FaceNet直接把输入图像变成欧式空间中的特征向量，两个特征向量间的欧式距离就可以用来衡量两者之间的相似度。
    # ori_imageEme = facenet(ori_image)
    # baseName = os.path.basename(oriImage_path)
    # save_dir = "abspath/Beauty_Algorithm/edited_Images/generation{}".format(generation)    
    pop_size = len(population)
    
    for idx in range(pop_size):
        logger.info(f"处理第{idx}个染色体！！！")
        logger.info(f"population[{idx}]={population[idx]}")
        # 编辑后的图像的保存路径
        os.makedirs(save_dir + "/{}".format(idx),exist_ok=True)
        save_path = save_dir + "/{}/{}-{}.png".format(idx,img.split("/")[-1].split(".")[0],idx)
        
        
        pro_imageEme = facenet(pro_images[idx])
        
        # 计算ssim值
        similarity = calculate_ssim(ori_image,pro_images[idx])
        o_Value.append(1-similarity)
        
        # 计算欧几里得距离。
        distance = torch.norm(ori_imageEme - pro_imageEme, p=2)
        logger.info(f"distance={distance}")
        o_Distance.append(distance)
        
        # 原始图像值进行一次编辑即可
        # if idx == 0 and generation == 0:
            # oriEditImage = edit_main(oriImage_path,os.path.join(save_dir,baseName),instruction)
                
        proEditImage = edit_main(proImage_paths[idx],save_path,instruction)
        
        proEditFeature = clip_model(save_path)
        # 计算clip相似度
        # Csimilarity = torch.norm(oriEditFeature-proEditFeature,p=2)
        # e_DValue.append(1-Csimilarity.cpu().numpy())
        Csimilarity = clip_img_score(oriImage_path,proImage_paths[idx])
        logger.info(f"Csimilarity={Csimilarity}")
        e_DValue.append(Csimilarity)
        
    
        oriEditImage_Eme = facenet(np.array(oriEditImage))
        proEditImage_Eme = facenet(np.array(proEditImage))
        
        similarity111 = calculate_ssim(np.array(oriEditImage),np.array(proEditImage))
        e_Value.append(similarity111)
        
        distance111 = torch.norm(oriEditImage_Eme - proEditImage_Eme, p=2)
        logger.info(f"distance111={distance111}")
        e_Distance.append(distance111)
        
        
        print(f"******************第{idx}张图片！！！***********************")
        print(f"******************绘制第{idx}个点!!!***********************")
        logger.info(f"编辑前图像的欧式距离，o_Distance={o_Distance[idx].item()}")
        logger.info(f"编辑后图像的欧式距离，e_Distance={e_Distance[idx].item()}")
        logger.info(f"编辑前后图像的欧式距离比例，Distance={o_Distance[idx].item()/e_Distance[idx].item()}")#希望o_Distance小，e_Distance大。
        logger.info(f"编辑前图像的ssim值，o_Value={o_Value[idx]}")
        logger.info(f"编辑后图像的ssim值，e_Value={e_Value[idx]}")
        logger.info(f"编辑前后图像的ssim值比例，Value={o_Value[idx]/e_Value[idx]}")# ssim值越大，两张图片越相似。希望o_Value小，e_Value大
        
        # fitness_values1.append(o_Value[idx])
        # fitness_values2.append(e_Value[idx])
        
        fitness_values1.append(o_Distance[idx].item())
        fitness_values2.append(1-e_Distance[idx].item())
        
        # fitness_values1.append(o_Distance[idx].item())
        # fitness_values2.append(e_DValue[idx])
    
    
    path = 'abspath/Beauty_Algorithm/fitness_values.txt'
    with open (path,"w") as file:
        file.write("generation_{}:".format(generation) + '\n')
        fitness1 = ",".join(map(str,fitness_values1))
        fitness2 = ",".join(map(str,fitness_values2))
        file.write("fitness1="+fitness1+'\n')
        file.write("fitness2="+fitness2+'\n')
        file.write("\n")
        
            
    plt.figure()   
    plt.scatter(fitness_values1,fitness_values2,s=20,marker='o')
    for idx in range(pop_size):
        plt.annotate(idx, xy=(fitness_values1[idx], fitness_values2[idx]), xytext=(fitness_values1[idx] , fitness_values2[idx]+ 0.002),fontsize=8)
        plt.text(fitness_values1[idx], fitness_values2[idx], f'{round(fitness_values1[idx],3)}, {round(fitness_values2[idx],3)}', fontsize=6, verticalalignment='top', horizontalalignment='center')

    plt.xlabel('function1')
    plt.ylabel('function2')
    plt.title('Generation {} Solution Distribution Diagram'.format(generation))
    plt.savefig('abspath/Beauty_Algorithm/fitness_images/fitness_function_{}.png'.format(generation))
    plt.show()
    plt.close()
    print(f'fitness_values1={fitness_values1}')
    print(f'fitness_values2={fitness_values2}')
    print(f'population={population}')
    print(f'population={len(population)}')
    
    
    return fitness_values1, fitness_values1



# 初始化种群
def init_population(pop_size, chromosome_length):
    #fixed_values = [None, None, None, None, 1, None, None]
    population = []
    
    # 生成种群
    for idx in range(pop_size):
        chromosome = [None] * chromosome_length
        # 确保第一位只能为5或者9
        #chromosome[0] = random.choice([5,9])
        chromosome[0] = random.randint(5,20)
        chromosome[1] = random.randint(0,255)
        chromosome[2] = random.randint(0,255)
        chromosome[3] = random.uniform(0.5,1)
        chromosome[4] = 1-chromosome[3]
        chromosome[5] = random.uniform(1,2)#锐化
        chromosome[6] = random.uniform(1,2)#对比度
        chromosome[7] = random.uniform(0.9,1.1)#饱和度
        chromosome[8] = random.uniform(0.9,1.1)#亮度

    
        population.append(chromosome)
    
    return population

# 交叉操作
def crossover(parent1, parent2, crossover_rate):
    if random.random() < crossover_rate:
        choice = random.choice(['012','56','78'])
        if choice == '012':
            child1 = parent1[:]
            child1[0:3] = parent2[0:3]
            child2 = parent2[:]
            child2[0:3] = parent1[0:3]
        if choice == '56':
            child1 = parent1[:]
            child1[5:7] = parent2[5:7]
            child2 = parent2[:]
            child2[5:7] = parent1[5:7]
        if choice == '78':
            child1 = parent1[:]
            child1[7:9] = parent2[7:9]
            child2 = parent2[:]
            child2[7:9] = parent1[7:9]
        return child1, child2

    else:
        return parent1, parent2

# 变异操作
def mutation(parent, mutation_rate):
    if random.random() < mutation_rate:

        parent[0] = random.randint(5,20)
        parent[1] = random.randint(0,255)
        parent[2] = random.randint(0,255)
        parent[3] = random.uniform(0.5,1)
        parent[4] = 1-parent[3]
        parent[5] = random.uniform(1,2)#锐化
        parent[6] = random.uniform(1,2)#对比度
        parent[7] = random.uniform(0.9,1.1)#饱和度
        parent[8] = random.uniform(0.9,1.1)#亮度
    return parent

# 交叉和变异:当前交叉变异作用的对象对全体染色体
def crossover_and_mutation(population,selected_individuals,crossover_rate,mutation_rate):
    # crossover_Offspring = []
    # mutation_Offspring = []
    offspring = []
    
    number = int(len(selected_individuals)/2)
    logger.info(f"交叉变异中选择的个体数量={number}")
    # 交叉
    for croIdx in range(number):
        parent1 = population[selected_individuals[2*croIdx]]
        parent2 = population[selected_individuals[2*croIdx+1]]
        child1, child2 = crossover(parent1, parent2, crossover_rate)
        offspring.append(child1)
        offspring.append(child2)
        
    # 变异
    for mutIdx in range(number):
        parent = offspring[mutIdx]
        child = mutation(parent,mutation_rate)
        offspring[mutIdx] = child
        #mutation_Offspring.append(child)
    
    return offspring
    

# 快速非支配排序
#values = [values1,values2] #多目标函数的适应度值组合
def fast_nondominated_sort(values):
    # 取目标函数1的解集
    values11 = values[0] 
    # 存放每个个体支配解的集合
    S = [[] for i in range(0,len(values11))] 
    # 存放每个群体的级别集合，一个级别对应一个[]
    front = [[]]
    # 存放每个个体被支配的解的个数，也就是由于其的解的个数
    n = [0 for i in range(0,len(values11))]
    # 存放每个个体的级别，无穷大代表级别最低，0代表级别最大，也就是最优。
    rank = [np.inf for i in range(0,len(values11))]
    
    # 遍历解集中的每一个个体，确定他们的支配与被支配关系
    for p in range(0,len(values11)):
        # 初始化存放参数的变量
        S[p] = []
        n[p] = 0
        for q in range(0,len(values11)):
            less = 0
            equal = 0
            greater = 0
            for k in range(len(values)):
                if values[k][p] > values[k][q]:
                    less = less + 1
                if values[k][p] == values[k][q]:
                    equal = equal +1
                if values[k][p] < values[k][q]:
                    greater = greater + 1
            # 记录其被支配的解的个数
            if (less + equal == len(values)) and (equal != len(values)):
                n[p] = n[p] + 1 
            # 记录其支配的解的集合
            elif (greater + equal == len(values)) and (equal != len(values)):
                S[p].append(q)
        # 找出Pareto最优解，也就是n[p]=0的个体的序号
        if n[p] == 0:
            rank[p] = 0
            if p not in front[0]:
                front[0].append(p)
    
    # 划分各层的解
    i = 0
    while(front[i] != []):
        Q = []
        # 遍历当前分层集合的各个个体p
        for p in front[i]:
            # 遍历p个体的每个支配解q
            for q in S[p]:
                # 遍历了个体p之后，将其支配解q的n[q]-1
                n[q] = n[q] -1
                # 当q个体的被支配解n[q]=0时，则意味着个体q为如今的最优解
                if n[q] == 0:
                    # 设置个体q的等级
                    rank[q] = i + 1
                    # 当q个体并不在已经记录的当前层的解中时，就将其添加进去
                    if q not in Q:
                        Q.append(q)
        i = i + 1
        front.append(Q)
    del front[len(front) - 1]
    
    return front

# 拥挤度计算                      
def crowding_distances(values,front,pop_size):
    distance = np.zeros(shape = (pop_size,))
    # 遍历Pareto前沿的每一层解
    for rank in front:
        # 遍历每一个解的每个目标函数值
        for i in range(len(values)): # i只取0或者1
            valuesi = [values[i][A] for A in rank]
            rank_valuesi = zip(rank,valuesi)
            # 先按函数值大小valuesi排序，如函数值大小相同，再按序号rank排序（均为升序排序）
            sort_rank_valuesi = sorted(rank_valuesi,key=lambda x:(x[1],x[0])) # sort_rank_valuesi=[(rank,xi/yi),...]
            sort_ranki = [j[0] for j in sort_rank_valuesi]
            sort_valuesi = [j[1] for j in sort_rank_valuesi]
            # 每一个rank中等级最优解和最差解的拥挤度距离设为inf
            distance[sort_ranki[0]] = np.inf
            distance[sort_ranki[-1]] = np.inf
            # 计算rank等级中，除去最优解和最差解之外，其余解的拥挤度距离
            for j in range(1,len(rank) - 2):
                if max(sort_valuesi) - min(sort_valuesi) != 0:
                    distance[sort_ranki[j]] = distance[sort_ranki[j]] + (sort_valuesi[j+1] - sort_valuesi[j-1]) / (max(sort_valuesi) - min(sort_valuesi))
                else:
                    distance[sort_ranki[j]] = np.inf
            
    distanceA = [[] for i in range(len(front))]
    for j in range(len(front)):
        for i in range(len(front[j])):
            distanceA[j].append(distance[front[j][i]])
    return distanceA,distance
    #return distanceA
            
# 选择
def select_individuals(front, distance, select_size):
    selected_individuals = []
    remaining_population_size = select_size
    
    # 遍历每个前沿
    for f in front:
        print(f'remaining_population_size={remaining_population_size}')
        print(f'distance={distance}')
        # 如果当前前沿中的个体数量不足以填满剩余种群空间，则将该前沿的所有个体都加入选择列表中
        #if len(selected_individuals) + len(f) <= select_size:
        if remaining_population_size == 0:
            break
        if len(selected_individuals) + len(f) <= select_size:
            selected_individuals.extend(f)
            remaining_population_size -= len(f)
        # 如果当前前沿中的个体数量超过了剩余种群空间，则根据拥挤度距离选择适当数量的个体
        else:
            # 根据拥挤度距离对当前前沿中的个体进行排序
            print(f"f={f}")
            f_sorted = sorted(f, key=lambda x: distance[x], reverse=True)
            print(f"f_sorted={f_sorted}")
            # 选择前n个个体填满剩余种群空间
            selected_individuals.extend(f_sorted[:remaining_population_size])
            break

    return selected_individuals

           
 
# 算法主体
def bnip(img, instruction, population_size, chromosome_length, max_generation, crossover_rate, mutation_rate):
    
    
    # 1.初始化种群
    logger.info(f"****************1.初始化种群*************************")
    population = init_population(population_size, chromosome_length)
    # 2.评估每一代的个体适应度
    logger.info(f"****************2.编辑图片并评估每一代的个体适应度*************************")
    #ori_image,pro_images,oriImage_path,proImage_paths = image_processing(img,population,generation)
    ori_imageEme = facenet(img)
    baseName = os.path.basename(img)
    save_dir111 = "abspath/Beauty_Algorithm/edited_Images"
    os.makedirs(save_dir111,exist_ok=True)
    oriImage_path = os.path.join(save_dir111,baseName)
    
    oriEditImage = edit_main(img,oriImage_path,instruction)
    oriEditFeature = clip_model(os.path.join(save_dir111,baseName))  
    file_path = "abspath/Beauty_Algorithm/population_data.txt"
    for generation in range(max_generation+1):
        # 保存每一轮的population
        #file_path = "abspath/Beauty_Algorithm/population_data_generation{}.txt".format(generation)
        with open(file_path,"w") as file:#"a"为追加模式，"w"为写模式
            for c_Idx,chromosome in enumerate(population):
                chromosome_data = ",".join(map(str,chromosome))
                file.write("number_{}:".format(generation)+"\n")
                file.write("chromosome" + str(c_Idx) + ":" + chromosome_data + "\n")
            file.write("\n")
        
        pop_size = len(population)
        select_size = population_size
        # 计算适应度
        save_dir = save_dir111 + '/generation{}'.format(generation)
        fitness_values1,fitness_values2 = fitness_function(img,oriImage_path,ori_imageEme,oriEditImage,oriEditFeature,population,generation,save_dir)
        logger.info(f"****************3.非支配排序*************************")
        # 3.非支配排序
        values = [fitness_values1,fitness_values2]
        front = fast_nondominated_sort(values)
        logger.info(f"****************4.计算拥挤度距离*************************")
        # 4.计算拥挤度距离
        distanceA,distance = crowding_distances(values,front,pop_size)
        logger.info(f"front={front}")
        logger.info(f"distanceA={distanceA}")
        logger.info(f"distance={distance}")
        logger.info(f"****************5.选择优秀的个体*************************")
        # 5.选择优秀的个体
        selected_individuals = select_individuals(front, distance, select_size)
        # 保存front，distance，selected_individuals信息到txt文件中
        filename = "abspath/Beauty_Algorithm/fron_distance_selected/front_distance_selected_{}.txt".format(generation)
        
        with open(filename, 'w') as f:
            # 保存 front
            f.write("Front:\n")
            f.write(','.join(map(str,front)))
            f.write('\n')
            for i, layer in enumerate(front):
                f.write(f"Layer {i}: {', '.join(map(str, layer))}\n")
            
            # 保存 distance
            f.write("\nDistance:\n")
            f.write(', '.join(map(str, distance)))
            
            # 保存 distanceA
            f.write("\nDistanceA:\n")
            for i, distances in enumerate(distanceA):
                f.write(f"Layer {i}: {', '.join(map(str, distances))}\n")

            # 保存 selected_individuals
            f.write("\nselected_individuals:\n")
            f.write(', '.join(map(str, selected_individuals)))

        
        logger.info(f"selected_individuals={selected_individuals}")
        logger.info(f"selected_individuals={len(selected_individuals)}")
        if generation == max_generation:
            logger.info(f'第{generation}代种群：population={selected_individuals}')
            logger.info(f'第{generation}代种群：population的数量={len(population)}')
            for idx in range(len(selected_individuals)):
                logger.info(f'第{generation}代种群：population={population[selected_individuals[idx]]}')
            logger.info(f"******************结束第{generation}代种群！！！****************************")
            return selected_individuals
        ### selected_individuals对应的是适应度值的序号也是染色体的序号，交叉变异得到的是染色体个体，也就是效果参数
        logger.info(f"****************6.交叉和变异*************************")
        # 6.交叉和变异
        offspring = crossover_and_mutation(population,selected_individuals,crossover_rate,mutation_rate)
        logger.info(f"****************7.种群更新,新旧两代合并*************************")
        # 7.种群更新,新旧两代合并
        combined_population = []
        for i in range(len(selected_individuals)):
            combined_population.append(population[selected_individuals[i]])
        logger.info(f'第{generation}代种群：combined_population={len(combined_population)}')
        combined_population.extend(offspring)
        population = combined_population
        logger.info(f'第{generation}代种群：population={population}')
        logger.info(f'第{generation}代种群：population的数量={len(population)}')
        logger.info(f"******************结束第{generation}代种群！！！****************************")


   
if __name__ == "__main__":
    
    
    # 输入的原始图片路径
    #img = "abspath/Beauty_Algorithm/20240409.png"
    img = "abspath/Beauty_Algorithm/20240422.png"
    instruction = "what would he look like as a bearded man?"

    random.seed(7)
    
    
    final_population = bnip(img,instruction,10,9,10,0.8,0.5)
    logger.info(f'final_population={final_population}')
