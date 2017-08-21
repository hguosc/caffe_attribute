#include <algorithm>
#include <vector>

#include "caffe/layers/maxratiodrop_layer.hpp"

namespace caffe {

template <typename Dtype>
void MaxratiodropLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
  /*threshold_ = this->layer_param_.dropout_param().dropout_ratio();
  DCHECK(threshold_ > 0.);
  DCHECK(threshold_ < 1.);
  scale_ = 1. / (1. - threshold_);
  uint_thres_ = static_cast<unsigned int>(UINT_MAX * threshold_);*/
}

template <typename Dtype>
void MaxratiodropLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::Reshape(bottom, top);
  // Set up the cache for random number generation
  // ReshapeLike does not work because rand_vec_ is of Dtype uint
  // rand_vec_.Reshape(bottom[0]->shape());
}

template <typename Dtype>
void MaxratiodropLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  const int num = bottom[0]->num();
  const int dim = count / num;
  Dtype max_ratio = this->layer_param_.maxratiodrop_param().max_ratio();
  for (int n = 0; n < num; ++n)
  {
    // find max for each sample
    Dtype sum_value = Dtype(0);
    for (int i = 0; i < dim; ++i)
    {
      Dtype value = bottom_data[n * dim + i];
      sum_value += value;
    }
    Dtype mean_value = sum_value / dim;
    Dtype threshold = mean_value * max_ratio;    // change from max ratio to times of mean
    threshold_.push_back(threshold);
    for (int i = 0; i < dim; ++i)
    {
      Dtype value = bottom_data[n * dim + i];
      int index = n*dim+i;
      if (value > threshold)
      {
        top_data[index] = bottom_data[index];
      }
      else
      {
        top_data[index] = Dtype(0);
      }
    }
  }
}

template <typename Dtype>
void MaxratiodropLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    const int num = bottom[0]->num();
    const int dim = count / num;
    for (int n = 0; n < num; ++n)
    {
      Dtype threshold = threshold_[n];
      for (int i = 0; i < dim; ++i)
      {
        Dtype value = bottom_data[n * dim + i];
        int index = n*dim+i;
        if (value > threshold)
        {
          bottom_diff[index] = top_diff[index];
        }
        else
        {
          bottom_diff[index] = Dtype(0);
        }
      }
    }

    /*if (this->phase_ == TRAIN) {
      const unsigned int* mask = rand_vec_.cpu_data();
      const int count = bottom[0]->count();
      for (int i = 0; i < count; ++i) {
        bottom_diff[i] = top_diff[i] * mask[i] * scale_;
      }
    } else {
      caffe_copy(top[0]->count(), top_diff, bottom_diff);
    }*/
  }
}


#ifdef CPU_ONLY
STUB_GPU(MaxratiodropLayer);
#endif

INSTANTIATE_CLASS(MaxratiodropLayer);
REGISTER_LAYER_CLASS(Maxratiodrop);

}  // namespace caffe
