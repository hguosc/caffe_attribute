#include <vector>

#include "caffe/layers/heatmap_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void HeatmapLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  has_ignore_label_ = this->layer_param_.loss_param().has_ignore_label();
  if (has_ignore_label_)
  {
    ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }
}

template <typename Dtype>
void HeatmapLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // LossLayer<Dtype>::Reshape(bottom, top);
  vector<int> bottom_shape = bottom[0]->shape();
  // channel should be 1
  // width == height
  vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
  top[0]->Reshape(loss_shape);
  result.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void HeatmapLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {  
  int count = bottom[0]->count();
  const float alpha = this->layer_param_.heatmap_loss_param().alpha();
  const float beta = this->layer_param_.heatmap_loss_param().beta();
  const int size = this->layer_param_.heatmap_loss_param().size();
  Dtype mean = 1 / Dtype(size * size); 
  Dtype * result_data = result.mutable_cpu_data();
  
  caffe_copy(count, bottom[0]->cpu_data(), result_data);
  
  caffe_add_scalar(count, (Dtype(0) - beta * mean), result_data); // x - b*m
  caffe_scal(count, Dtype(0-alpha), result_data);     // a(x - bm)
  caffe_exp(count, result_data, result_data);
  Dtype loss_total = Dtype(0);
  int loss_count = 0;
  if (bottom.size() > 1)
  {
    const Dtype * label = bottom[1]->cpu_data();
    for (int i = 0; i < count; ++i)
    {
      const int label_value = static_cast<int>(label[i]);
      if (has_ignore_label_ && label_value == ignore_label_)
      {
        continue;
      }
      loss_total += result_data[i];
      ++loss_count;
    }
  }
  // top[0]->mutable_cpu_data()[0] = caffe_cpu_asum(count, result_data) / count;
  top[0]->mutable_cpu_data()[0] = loss_total / loss_count;
}

template <typename Dtype>
void HeatmapLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  // const Dtype * top_data = top[0]->cpu_data();
  const Dtype * top_diff = top[0]->cpu_diff();
  Dtype * bottom_diff = bottom[0]->mutable_cpu_diff();
  int count = bottom[0]->count();
  const float alpha = this->layer_param_.heatmap_loss_param().alpha();
  const float beta = this->layer_param_.heatmap_loss_param().beta();
  const int size = this->layer_param_.heatmap_loss_param().size();
  Dtype mean = 1 / Dtype(size * size);
  
  caffe_copy(count, bottom[0]->cpu_data(), bottom_diff);
  caffe_add_scalar(count, (Dtype(0) - beta * mean), bottom_diff); // x - b*m
  caffe_scal(count, Dtype(0-alpha), bottom_diff);     // a(x - bm)
  caffe_exp(count, bottom_diff, bottom_diff);
  caffe_scal(count, Dtype(0-alpha/count), bottom_diff);
  caffe_mul(count, top_diff, bottom_diff, bottom_diff);
  if (bottom.size() > 1)
  {
    const Dtype * label = bottom[1]->cpu_data();
    for (int i = 0; i < count; ++i)
    {
      const int label_value = static_cast<int>(label[i]);
      if (has_ignore_label_ && label_value == ignore_label_)
      {
        bottom_diff[i] = 0;
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(HeatmapLossLayer);
#endif

INSTANTIATE_CLASS(HeatmapLossLayer);
REGISTER_LAYER_CLASS(HeatmapLoss);

}  // namespace caffe
