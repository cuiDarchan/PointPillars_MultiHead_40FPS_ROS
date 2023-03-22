/******************************************************************************
 * Copyright 2020 The Apollo Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *****************************************************************************/

/*
 * Copyright 2018-2019 Autoware Foundation. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @author Kosuke Murakami
 * @date 2019/02/26
 */

/**
* @author Yan haixu
* Contact: just github.com/hova88
* @date 2021/04/30
*/



// headers in STL
#include <stdio.h>

// headers in local files
#include "common.h"
#include "preprocess.h"

/*
 * pillar化：根据pillar坐标，1）为pillar_point_feature的赋值，2）统计每一个pillar中点数
 */
__global__ void make_pillar_histo_kernel(
    const float* dev_points, float* dev_pillar_point_feature_in_coors,
    int* pillar_count_histo, const int num_points,
    const int max_points_per_pillar, const int grid_x_size,
    const int grid_y_size, const int grid_z_size, const float min_x_range,
    const float min_y_range, const float min_z_range, const float pillar_x_size,
    const float pillar_y_size, const float pillar_z_size,
    const int num_point_feature) {
  int th_i = blockIdx.x * blockDim.x +  threadIdx.x ;
  if (th_i >= num_points) {
    return;
  }
  int x_coor = floor((dev_points[th_i * num_point_feature + 0] - min_x_range) / pillar_x_size);
  int y_coor = floor((dev_points[th_i * num_point_feature + 1] - min_y_range) / pillar_y_size);
  int z_coor = floor((dev_points[th_i * num_point_feature + 2] - min_z_range) / pillar_z_size);

  if (x_coor >= 0 && x_coor < grid_x_size && y_coor >= 0 &&
      y_coor < grid_y_size && z_coor >= 0 && z_coor < grid_z_size) {
    int count =
        atomicAdd(&pillar_count_histo[y_coor * grid_x_size + x_coor], 1); //先赋值后进行加法计算, 统计pillar中点数count
    if (count < max_points_per_pillar) {
      // 索引[y_coor, x_coor， max_points_per_pillar, num_point_feature]变换为pillar索引
      // max_points_per_pillar: N, num_point_feature: D(9)， ind: 每一个点对应一个pillar索引
      int ind =
          y_coor * grid_x_size * max_points_per_pillar * num_point_feature +
          x_coor * max_points_per_pillar * num_point_feature +
          count * num_point_feature;
      // 将每一个点特征放入一个pillar中
      for (int i = 0; i < num_point_feature; ++i) {
        dev_pillar_point_feature_in_coors[ind + i] =
            dev_points[th_i * num_point_feature + i];// dev_pillar_point_feature_in_coors 一维
      }
    }
  }
}


/*
 * 对每一个pillar进行操作： 1）统计pillar中点的个数更新；2）生成pillar索引；3）生成带有占据信息的sparse_pillar_map；4）保存pillar坐标
 */
__global__ void make_pillar_index_kernel(
    int* dev_pillar_count_histo, int* dev_counter, int* dev_pillar_count,
    int* dev_x_coors, int* dev_y_coors, float* dev_num_points_per_pillar,
    int* dev_sparse_pillar_map, const int max_pillars,
    const int max_points_per_pillar, const int grid_x_size,
    const int num_inds_for_scan) {
  int x = blockIdx.x; // grid_x_size
  int y = threadIdx.x; // grid_y_size
  int num_points_at_this_pillar = dev_pillar_count_histo[y * grid_x_size + x];// pillar中点数
  // pillar中没有点则过滤掉
  if (num_points_at_this_pillar == 0) {
    return;
  }

  int count = atomicAdd(dev_counter, 1); // dev_counter: 统计 pillar数目
  if (count < max_pillars) { // 30000
    atomicAdd(dev_pillar_count, 1); // dev_pillar_count: 统计未超过最大pillar数目的 非空pillar数目
    // 若某个pillar中点数超限，则将其中的点数赋予max_points（20）
    if (num_points_at_this_pillar >= max_points_per_pillar) {
      dev_num_points_per_pillar[count] = max_points_per_pillar;  // 20
    } else {
      dev_num_points_per_pillar[count] = num_points_at_this_pillar;
    }
    dev_x_coors[count] = x; // blockIdx.x，记录线程坐标, 也是pillar坐标，一个pillar对应一个thread
    dev_y_coors[count] = y; // threadIdx.x, (x,y)，绘制一维block行，thread列帮助理解
    dev_sparse_pillar_map[y * num_inds_for_scan + x] = 1;//标记sparse_pillar_map 占据为1， num_inds_for_scan：1024, 等效于wh中w
  }
}

/*
 * 构建 pillar_feature, 对pillar中的每一个点执行如下操作
 */
__global__ void make_pillar_feature_kernel(
    float* dev_pillar_point_feature_in_coors, float* dev_pillar_point_feature,
    float* dev_pillar_coors, int* dev_x_coors, int* dev_y_coors,
    float* dev_num_points_per_pillar, const int max_points,
    const int num_point_feature, const int grid_x_size) {
  int ith_pillar = blockIdx.x;// pillar索引
  int num_points_at_this_pillar = dev_num_points_per_pillar[ith_pillar];
  int ith_point = threadIdx.x;// pillar中每一个点执行一个thread
  if (ith_point >= num_points_at_this_pillar) {
    return;
  }
  int x_ind = dev_x_coors[ith_pillar];
  int y_ind = dev_y_coors[ith_pillar];
  // 注意： pillar_ind 并非一个pillar一个ind，而是一个pillar中的每一个点对应一个pillar_idx
  int pillar_ind = ith_pillar * max_points * num_point_feature +
                   ith_point * num_point_feature;    //（P,N,D）
  int coors_ind = y_ind * grid_x_size * max_points * num_point_feature +
                  x_ind * max_points * num_point_feature +
                  ith_point * num_point_feature;     // (w,h,N,D)
  // pillar_point_feature_in_coors 转换为 pillar_point_feature
  #pragma unroll //优化：展开小循环，提高运行效率
  for (int i = 0; i < num_point_feature; ++i) { // 5 ,x y, z ,i ,0
    dev_pillar_point_feature[pillar_ind + i] =
        dev_pillar_point_feature_in_coors[coors_ind + i];
  }

  float coor_x = static_cast<float>(x_ind);
  float coor_y = static_cast<float>(y_ind);
  dev_pillar_coors[ith_pillar * 4 + 0] = 0;  // batch idx
  dev_pillar_coors[ith_pillar * 4 + 1] = 0;  // z
  dev_pillar_coors[ith_pillar * 4 + 2] = coor_y;
  dev_pillar_coors[ith_pillar * 4 + 3] = coor_x;
}


/*
 * 构建均值 dev_points_mean，每一个thread对应一个点的某一个维度，如x
 * 注意： 二维block索引坐标，轴向右x,下y，与图像相同。
 */
__global__ void pillar_mean_kernel(
  float* dev_points_mean, 
  const int num_point_feature,
  const float* dev_pillar_point_feature, 
  const float* dev_num_points_per_pillar, 
  int max_pillars , 
  int max_points_per_pillar) {

    // 一个block对应一个共享内存，一个grid对应一个全局内存global memory，
    extern __shared__ float temp[]; // 一个点，对应一个block，对应一个共享内存
    int ith_pillar = blockIdx.x; 
    int ith_point  = threadIdx.x; //thread 二维，（20,3）
    int axis = threadIdx.y; // 为求xyz三个维度算数平均值，维度是3
  
    int reduce_size = max_points_per_pillar > 32 ? 64 : 32; // 32
    // printf("reduce_size: %d\n", reduce_size); // 32
    
    /// temp数组代表32个点，先提前都赋值上，后续根据pillar中真实点数, 进行平均值求取，不足点数的补0
    // temp数组前半部分： [x0,y0,z0, x1,y1,z1, ... x19,y19,z19]
    temp[threadIdx.x * 3 + axis] =  dev_pillar_point_feature[ith_pillar * max_points_per_pillar * num_point_feature + ith_point * num_point_feature + axis];  //只取前20个，剩余12个直接补0
    // temp数组后半部分： [x20,y20,z20, ... x31,y31,z31] 全部赋值为0
    if (threadIdx.x < reduce_size - max_points_per_pillar) { // 12，pillar中其余值补0
        temp[(threadIdx.x + max_points_per_pillar) * 3 + axis] = 0.0f; //--> dummy placeholds will set as 0
    }
    __syncthreads(); //block级别线程同步
    int num_points_at_this_pillar = dev_num_points_per_pillar[ith_pillar];

    if (ith_point >= num_points_at_this_pillar) {
          return;
    }

    // reduce_size = 32 , 二进制 10000， 循环五次，d取值为16,8,4,2，1
    for (unsigned int d = reduce_size >> 1 ; d > 0.6; d >>= 1) {
        if (ith_point < d) {
            temp[ith_point*3 +axis] += temp[(ith_point + d) * 3 + axis];// 折半累加，后16个加到前16个上，然后前16个分两半，后8个加到前8个上，以此类推
        }
        // printf("pillar_mean_kernel, d: %d\n",d);
        __syncthreads();
    }

    // 累加后，第一个点就xyz就是所有点xyz的累加和
    if (ith_point == 0) {
        dev_points_mean[ith_pillar * 3 + axis] = temp[ith_point + axis] / num_points_at_this_pillar ;
        // printf("dev_points_mean, d: %d\n",num_points_at_this_pillar);
    }
}
















__device__ void warpReduce(volatile float* sdata , int ith_point , int axis) {
    sdata[ith_point * blockDim.y + axis] += sdata[(ith_point + 8) * blockDim.y + axis];
    sdata[ith_point * blockDim.y + axis] += sdata[(ith_point + 4) * blockDim.y + axis];
    sdata[ith_point * blockDim.y + axis] += sdata[(ith_point + 2) * blockDim.y + axis];
    sdata[ith_point * blockDim.y + axis] += sdata[(ith_point + 1) * blockDim.y + axis];
}





__global__ void make_pillar_mean_kernel(
  float* dev_points_mean, 
  const int num_point_feature,
  const float* dev_pillar_point_feature, 
  const float* dev_num_points_per_pillar, 
  int max_pillars , 
  int max_points_pre_pillar) {
    extern __shared__ float temp[];
    unsigned int ith_pillar = blockIdx.x;  // { 0 , 1, 2, ... , 10000+}
    unsigned int ith_point  = threadIdx.x; // { 0 , 1, 2, ...,9}
    unsigned int axis = threadIdx.y; 
    unsigned int idx_pre  = ith_pillar * max_points_pre_pillar * num_point_feature \
                     + ith_point  * num_point_feature;
    unsigned int idx_post = ith_pillar * max_points_pre_pillar * num_point_feature \
                     + (ith_point + blockDim.x)  * num_point_feature;

    temp[ith_point * blockDim.y + axis] = 0.0;
    unsigned int num_points_at_this_pillar = dev_num_points_per_pillar[ith_pillar];

    // if (ith_point < num_points_at_this_pillar / 2) {
      temp[ith_point * blockDim.y + axis] = dev_pillar_point_feature[idx_pre  + axis] 
                                          + dev_pillar_point_feature[idx_post + axis];
    // }
    __syncthreads();//block内部用于线程同步

    // do reduction in shared mem
    // Sequential addressing. This solves the bank conflicts as
    // the threads now access shared memory with a stride of one
    // 32-bit word (unsigned int) now, which does not cause bank 
    // conflicts
    warpReduce(temp , ith_point , axis);

	// // write result for this block to global mem
    if (ith_point == 0)
    dev_points_mean[ith_pillar * blockDim.y + axis] = temp[ith_point * blockDim.y + axis] / num_points_at_this_pillar ;
}


/*
 * 聚合所有得到的 point_feature，每一个thread对应一个点
 */
__global__ void gather_point_feature_kernel(
  const int max_num_pillars_,const int max_num_points_per_pillar,const int num_point_feature,
  const float min_x_range, const float min_y_range, const float min_z_range, 
  const float pillar_x_size,  const float pillar_y_size, const float pillar_z_size,
  const float* dev_pillar_point_feature, const float* dev_num_points_per_pillar, 
  const float* dev_pillar_coors,
  float* dev_points_mean, 
  float* dev_pfe_gather_feature_){

  int ith_pillar = blockIdx.x; 
  int ith_point = threadIdx.x;
  // int kNumPointFeature = 5;
  int num_gather_feature = 11;
  int num_points_at_this_pillar = dev_num_points_per_pillar[ith_pillar];

  // 若线程id超过 pillar中的点数，则直接返回
  if (ith_point >= num_points_at_this_pillar){
        return;
    }

    // 11 维特征， x, y, z ,i, 0, 归一化cx,cy,cz, 到pillar偏移 xp,yp,zp
    dev_pfe_gather_feature_[ith_pillar * max_num_points_per_pillar * num_gather_feature + ith_point * num_gather_feature + 0] 
    =  dev_pillar_point_feature[ith_pillar * max_num_points_per_pillar * num_point_feature + ith_point * num_point_feature + 0]; 
  
    dev_pfe_gather_feature_[ith_pillar * max_num_points_per_pillar * num_gather_feature + ith_point * num_gather_feature + 1]  
    =  dev_pillar_point_feature[ith_pillar * max_num_points_per_pillar * num_point_feature + ith_point * num_point_feature + 1];
  
    dev_pfe_gather_feature_[ith_pillar * max_num_points_per_pillar * num_gather_feature + ith_point * num_gather_feature + 2]  
    =  dev_pillar_point_feature[ith_pillar * max_num_points_per_pillar * num_point_feature + ith_point * num_point_feature + 2];
  
    dev_pfe_gather_feature_[ith_pillar * max_num_points_per_pillar * num_gather_feature + ith_point * num_gather_feature + 3]  
    =  dev_pillar_point_feature[ith_pillar * max_num_points_per_pillar * num_point_feature + ith_point * num_point_feature + 3];

    dev_pfe_gather_feature_[ith_pillar * max_num_points_per_pillar * num_gather_feature + ith_point * num_gather_feature + 4]  
    =  dev_pillar_point_feature[ith_pillar * max_num_points_per_pillar * num_point_feature + ith_point * num_point_feature + 4];
  
    // dev_pfe_gather_feature_[ith_pillar * max_num_points_per_pillar * num_gather_feature + ith_point * num_gather_feature + 4]  =  0.0f;
    //   f_cluster = voxel_features[:, :, :3] - points_mean
    dev_pfe_gather_feature_[ith_pillar * max_num_points_per_pillar * num_gather_feature + ith_point * num_gather_feature + 5]  
    =  dev_pillar_point_feature[ith_pillar * max_num_points_per_pillar * num_point_feature + ith_point * num_point_feature + 0] - dev_points_mean[ith_pillar * 3 + 0 ];

    dev_pfe_gather_feature_[ith_pillar * max_num_points_per_pillar * num_gather_feature + ith_point * num_gather_feature + 6] 
    =  dev_pillar_point_feature[ith_pillar * max_num_points_per_pillar * num_point_feature + ith_point * num_point_feature + 1] - dev_points_mean[ith_pillar * 3 + 1 ];
  
    dev_pfe_gather_feature_[ith_pillar * max_num_points_per_pillar * num_gather_feature + ith_point * num_gather_feature + 7]  
    =  dev_pillar_point_feature[ith_pillar * max_num_points_per_pillar * num_point_feature + ith_point * num_point_feature + 2] - dev_points_mean[ith_pillar * 3 + 2 ];

    // f_center[:, :, 0] = voxel_features[:, :, 0] - (coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
    dev_pfe_gather_feature_[ith_pillar * max_num_points_per_pillar * num_gather_feature + ith_point * num_gather_feature + 8]  
    =  dev_pillar_point_feature[ith_pillar * max_num_points_per_pillar * num_point_feature + ith_point * num_point_feature + 0] - (dev_pillar_coors[ith_pillar * 4 + 3] * pillar_x_size + (pillar_x_size/2 + min_x_range)); // min_x_range 最小范围值 -51.2
  
    dev_pfe_gather_feature_[ith_pillar * max_num_points_per_pillar * num_gather_feature + ith_point * num_gather_feature + 9]  
    =  dev_pillar_point_feature[ith_pillar * max_num_points_per_pillar * num_point_feature + ith_point * num_point_feature + 1] - (dev_pillar_coors[ith_pillar * 4 + 2] * pillar_y_size + (pillar_y_size/2 + min_y_range));
  
    dev_pfe_gather_feature_[ith_pillar * max_num_points_per_pillar * num_gather_feature + ith_point * num_gather_feature + 10] 
    =  dev_pillar_point_feature[ith_pillar * max_num_points_per_pillar * num_point_feature + ith_point * num_point_feature + 2] - (dev_pillar_coors[ith_pillar * 4 + 1] * pillar_z_size + (pillar_z_size/2 + min_z_range));

}




PreprocessPointsCuda::PreprocessPointsCuda(
    const int num_threads, const int max_num_pillars,
    const int max_points_per_pillar, const int num_point_feature,
    const int num_inds_for_scan, const int grid_x_size, const int grid_y_size,
    const int grid_z_size, const float pillar_x_size, const float pillar_y_size,
    const float pillar_z_size, const float min_x_range, const float min_y_range,
    const float min_z_range)
    : num_threads_(num_threads),
      max_num_pillars_(max_num_pillars),
      max_num_points_per_pillar_(max_points_per_pillar),
      num_point_feature_(num_point_feature), // 5
      num_inds_for_scan_(num_inds_for_scan),
      grid_x_size_(grid_x_size),
      grid_y_size_(grid_y_size),
      grid_z_size_(grid_z_size),
      pillar_x_size_(pillar_x_size),
      pillar_y_size_(pillar_y_size),
      pillar_z_size_(pillar_z_size),
      min_x_range_(min_x_range),
      min_y_range_(min_y_range),
      min_z_range_(min_z_range) {
    
    GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_pillar_point_feature_in_coors_),
        grid_y_size_ * grid_x_size_ * max_num_points_per_pillar_ *  num_point_feature_ * sizeof(float)));
    GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_pillar_count_histo_),
        grid_y_size_ * grid_x_size_ * sizeof(int)));
    GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_counter_), sizeof(int)));
    GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_pillar_count_), sizeof(int)));    
    GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_points_mean_), max_num_pillars_ * 3 *sizeof(float)));  
    }

PreprocessPointsCuda::~PreprocessPointsCuda() {
    GPU_CHECK(cudaFree(dev_pillar_point_feature_in_coors_));
    GPU_CHECK(cudaFree(dev_pillar_count_histo_));
    GPU_CHECK(cudaFree(dev_counter_));
    GPU_CHECK(cudaFree(dev_pillar_count_));
    GPU_CHECK(cudaFree(dev_points_mean_));
  }


void PreprocessPointsCuda::DoPreprocessPointsCuda(
    const float* dev_points, const int in_num_points, 
    int* dev_x_coors,int* dev_y_coors, 
    float* dev_num_points_per_pillar,
    float* dev_pillar_point_feature, float* dev_pillar_coors,
    int* dev_sparse_pillar_map, int* host_pillar_count , float* dev_pfe_gather_feature) {
    // initialize paraments
    GPU_CHECK(cudaMemset(dev_pillar_point_feature_in_coors_, 0 , grid_y_size_ * grid_x_size_ * max_num_points_per_pillar_ *  num_point_feature_ * sizeof(float)));
    GPU_CHECK(cudaMemset(dev_pillar_count_histo_, 0 , grid_y_size_ * grid_x_size_ * sizeof(int)));
    GPU_CHECK(cudaMemset(dev_counter_, 0, sizeof(int)));
    GPU_CHECK(cudaMemset(dev_pillar_count_, 0, sizeof(int)));
    GPU_CHECK(cudaMemset(dev_points_mean_, 0,  max_num_pillars_ * 3 * sizeof(float)));
    int num_block = DIVUP(in_num_points , num_threads_); // 除法向上取整，num_threads = 64
    std::cout << "num_block: " << num_block << ", num_threads_:" << num_threads_ << std::endl; // 4188， 64
    make_pillar_histo_kernel<<<num_block , num_threads_>>>(
        dev_points, dev_pillar_point_feature_in_coors_, dev_pillar_count_histo_,
        in_num_points, max_num_points_per_pillar_, grid_x_size_, grid_y_size_,
        grid_z_size_, min_x_range_, min_y_range_, min_z_range_, pillar_x_size_,
        pillar_y_size_, pillar_z_size_, num_point_feature_);
    
    // 每一个pillar中执行该操作
    make_pillar_index_kernel<<<grid_x_size_, grid_y_size_>>>(
        dev_pillar_count_histo_, dev_counter_, dev_pillar_count_, dev_x_coors,
        dev_y_coors, dev_num_points_per_pillar, dev_sparse_pillar_map,
        max_num_pillars_, max_num_points_per_pillar_, grid_x_size_,
        num_inds_for_scan_);  

    GPU_CHECK(cudaMemcpy(host_pillar_count, dev_pillar_count_, 1 * sizeof(int),
        cudaMemcpyDeviceToHost));
    std::cout << "host_pillar_count: " << *host_pillar_count << std::endl; // 28395
    make_pillar_feature_kernel<<<host_pillar_count[0],max_num_points_per_pillar_>>>(
        dev_pillar_point_feature_in_coors_, dev_pillar_point_feature,
        dev_pillar_coors, dev_x_coors, dev_y_coors, dev_num_points_per_pillar,
        max_num_points_per_pillar_, num_point_feature_, grid_x_size_);
    

    dim3 mean_block(max_num_points_per_pillar_,3); //(20,3)

    // pillar_mean_kernel第三个参数指定的是 shared memory大小
    pillar_mean_kernel<<<host_pillar_count[0],mean_block, 64 * 3 *sizeof(float)>>>(
      dev_points_mean_  ,num_point_feature_, dev_pillar_point_feature, dev_num_points_per_pillar, 
        max_num_pillars_ , max_num_points_per_pillar_);

    // dim3 mean_block(10,3); // Unrolling the Last Warp
    // make_pillar_mean_kernel<<<host_pillar_count[0], mean_block , 32 * 3 *sizeof(float)>>>(
    //       dev_points_mean_  ,num_point_feature_, dev_pillar_point_feature, dev_num_points_per_pillar, 
    //       max_num_pillars_ , max_num_points_per_pillar_);

    gather_point_feature_kernel<<<max_num_pillars_, max_num_points_per_pillar_>>>(
      max_num_pillars_,max_num_points_per_pillar_,num_point_feature_,
      min_x_range_, min_y_range_, min_z_range_,
      pillar_x_size_, pillar_y_size_, pillar_z_size_, 
      dev_pillar_point_feature, dev_num_points_per_pillar, dev_pillar_coors,
      dev_points_mean_,
      dev_pfe_gather_feature);

    // DEVICE_SAVE<float>(dev_pillar_point_feature , \
    //     max_num_pillars_ * max_num_points_per_pillar_ * num_point_feature_ , "dev_pillar_point_feature");
    // DEVICE_SAVE<float>(dev_num_points_per_pillar , \
    //   max_num_pillars_ , "dev_num_points_per_pillar");
    // DEVICE_SAVE<float>(dev_pfe_gather_feature , \
    //   max_num_pillars_ * 11, "dev_pfe_gather_feature");
}


