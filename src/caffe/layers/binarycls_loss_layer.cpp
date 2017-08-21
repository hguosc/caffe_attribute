#include <vector>

#include "caffe/layers/binarycls_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void BinaryclsLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  result.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void BinaryclsLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  // vector<int> b_shape = bottom[0]->shape(); // bottom should be the shape of N * C * 1 *1
  // int num = b_shape[0];
  // int channel = b_shape[1];
  const Dtype * bottom_data_1 = bottom[0]->cpu_data();  // scores
  const Dtype * bottom_data_2 = bottom[1]->cpu_data();  // labels, the sequence of scores and labels can not be changed
  Dtype * result_data = result.mutable_cpu_data();

  caffe_copy(count, bottom_data_2, result_data);  // y
  caffe_scal(count, Dtype(-2), result_data);      // -2*y
  caffe_add_scalar(count, Dtype(1), result_data); // -2y + 1
  caffe_mul(count, bottom_data_1, result_data, result_data);  // result_data is not const, see if it works // (1-2y)*x
  caffe_exp(count, result_data, result_data);     // exp{(1-2y)*x}
  caffe_add_scalar(count, Dtype(1), result_data); // 1 + exp{(1-2y)*x}
  caffe_log(count, result_data, result_data);     // ln(...)
  top[0]->mutable_cpu_data()[0] = caffe_cpu_asum(count, result_data);
}

template <typename Dtype>
void BinaryclsLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  // const Dtype * top_data = top[0]->cpu_data();
  const Dtype * top_diff = top[0]->cpu_diff();
  const Dtype * bottom_data_1 = bottom[0]->cpu_data();
  const Dtype * bottom_data_2 = bottom[1]->cpu_data();
  Dtype * bottom_diff = bottom[0]->mutable_cpu_diff();
  const Dtype scale = top_diff[0];
  int count = bottom[0]->count();

  Dtype * exp_data = new Dtype[count];
  Dtype * frac_1 = new Dtype[count];
  Dtype * frac_2 = new Dtype[count];
  caffe_memset(count, Dtype(0), exp_data);
  caffe_memset(count, Dtype(0), frac_1);
  caffe_memset(count, Dtype(0), frac_2);
  caffe_copy(count, bottom_data_2, exp_data);  // y
  caffe_scal(count, Dtype(-2), exp_data);      // -2*y
  caffe_add_scalar(count, Dtype(1), exp_data); // -2y + 1

  caffe_copy(count, exp_data, frac_1);  // frac_1 = (1-2y)
  caffe_mul(count, bottom_data_1, exp_data, exp_data);  // exp_data is not const, see if it works // (1-2y)*x
  caffe_exp(count, exp_data, exp_data);     // exp{(1-2y)*x}
  // frac_1
  caffe_mul(count, exp_data, frac_1, frac_1);

  // caffe_add_scalar(count, Dtype(1), exp_data); // 1 + exp{(1-2y)*x}
  // frac_2
  caffe_copy(count, exp_data, frac_2);
  caffe_add_scalar(count, Dtype(1), frac_2);
  caffe_div(count, frac_1, frac_2, bottom_diff);
  /*for (int i = 0; i < count; ++i)
  {
    Dtype v1 = frac_1[i];
    Dtype v2 = frac_2[i];
    LOG(INFO) << "exp_data [" << i <<"] = " << exp_data[i];
    LOG(INFO) << "gradient = " << v1 / v2;
  }*/

  caffe_scal(count, scale, bottom_diff);

  delete [] exp_data;
  delete [] frac_1;
  delete [] frac_2;
}

#ifdef CPU_ONLY
STUB_GPU(BinaryclsLossLayer);
#endif

INSTANTIATE_CLASS(BinaryclsLossLayer);
REGISTER_LAYER_CLASS(BinaryclsLoss);

}  // namespace caffe
