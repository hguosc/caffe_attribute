#include <vector>

#include "caffe/layers/sigmadjust_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SigmAdjustLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int count = top[0]->count();
  Dtype* top_data = top[0]->mutable_gpu_data();
  // caffe_gpu_abs(count, bottom[0]->gpu_data(), top_data);
  caffe_copy(count, bottom[0]->cpu_data(), top_data);
  caffe_gpu_scal(count, Dtype(2), top_data);  // 2*x
  caffe_gpu_add_scalar(count, Dtype(-1), top_data); // 2*x - 1
}

template <typename Dtype>
void SigmAdjustLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const int count = top[0]->count();
  const Dtype* top_diff = top[0]->gpu_diff();
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_copy(count, top_diff, bottom_diff);
    caffe_gpu_scal(count, Dtype(2), bottom_diff);
    // const Dtype* bottom_data = bottom[0]->gpu_data();
    // Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    // caffe_gpu_sign(count, bottom_data, bottom_diff);
    // caffe_gpu_mul(count, bottom_diff, top_diff, bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SigmAdjustLayer);


}  // namespace caffe
