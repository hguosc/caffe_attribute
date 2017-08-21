#include <vector>

#include "caffe/layers/sigmadjust_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SigmAdjustLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
  CHECK_NE(top[0], bottom[0]) << this->type() << " Layer does not "
    "allow in-place computation.";
}

template <typename Dtype>
void SigmAdjustLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int count = top[0]->count();
  Dtype* top_data = top[0]->mutable_cpu_data();
  // caffe_abs(count, bottom[0]->cpu_data(), top_data);
  caffe_copy(count, bottom[0]->cpu_data(), top_data);
  caffe_scal(count, Dtype(2), top_data);  // 2*x
  caffe_add_scalar(count, Dtype(-1), top_data); // 2*x - 1
}

template <typename Dtype>
void SigmAdjustLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const int count = top[0]->count();
  const Dtype* top_diff = top[0]->cpu_diff();
  if (propagate_down[0]) {
    // const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_copy(count, top_diff, bottom_diff);
    caffe_scal(count, Dtype(2), bottom_diff);
    
    // const Dtype* bottom_data = bottom[0]->cpu_data();
    // Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    // caffe_cpu_sign(count, bottom_data, bottom_diff);
    // caffe_mul(count, bottom_diff, top_diff, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(SigmAdjustLayer);
#endif

INSTANTIATE_CLASS(SigmAdjustLayer);
REGISTER_LAYER_CLASS(SigmAdjust);

}  // namespace caffe
