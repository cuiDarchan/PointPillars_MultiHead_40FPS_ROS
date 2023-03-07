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



//headers in local files
#include "scatter.h"

/*
 * scatter算子：将特征(C,P)还原回scattered_feature（W,H,C）, 每一个pillar中每一个特征执行该算子
 * 注意：索引计算技巧，如针对tensor（x,y,z） ind = x_size* y_size * z_i + x_size * y_i + x_i
 */
__global__ void scatter_kernel(int *x_coors, int *y_coors, float *pfe_output,
  float *scattered_feature, const int grid_x_size,
  const int grid_y_size) {
int i_pillar = blockIdx.x;
int i_feature = threadIdx.x;
int x_ind = x_coors[i_pillar]; // W
int y_ind = y_coors[i_pillar]; // H
float feature = pfe_output[i_pillar * 64 + i_feature];
scattered_feature[i_feature * grid_y_size * grid_x_size +
y_ind * grid_x_size + x_ind] = feature;
}

ScatterCuda::ScatterCuda(const int num_threads, const int grid_x_size,
const int grid_y_size)
: num_threads_(num_threads),
grid_x_size_(grid_x_size),
grid_y_size_(grid_y_size) {}

void ScatterCuda::DoScatterCuda(const int pillar_count, int *x_coors,
   int *y_coors, float *pfe_output,
   float *scattered_feature) {
  // num_threads_=C : 64
scatter_kernel<<<pillar_count, num_threads_>>>(x_coors, y_coors, pfe_output,
                    scattered_feature,
                    grid_x_size_, grid_y_size_);
}
