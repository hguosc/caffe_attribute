#include <cfloat>
#include <vector>

#include "caffe/layers/weighted_average_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void WeightedAveLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  LOG(INFO) << "weighted_average_layer layersetup";
  // the # of channels in bottom[0] should be equal to the second dimention of bottom[1];
  // only support 2 bottoms currently
  vector<int> featmap_shape = bottom[0]->shape();
  vector<int> weights_shape = bottom[1]->shape();
  LOG(INFO) << "featmap_shape: " << featmap_shape[0] << ", " << featmap_shape[1] << ", " << featmap_shape[2] <<","<< featmap_shape[3];
  LOG(INFO) << "weights_shape: " << weights_shape[0] << ", " << weights_shape[1];
  CHECK(featmap_shape[1] == weights_shape[1]) << "the # of channels in bottom[0] should be equal to the second dimention of bottom[1]";
  // selected_class = this->layer_param_.weighted_ave_param().selected_class();
}

template <typename Dtype>
void WeightedAveLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  vector<int> featmap_shape = bottom[0]->shape();
  vector<int> weights_shape = bottom[1]->shape();
  vector<int> top_shape;
  top_shape.push_back(featmap_shape[0]);
  top_shape.push_back(featmap_shape[2]);
  top_shape.push_back(featmap_shape[3]);
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void WeightedAveLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* feat_data = bottom[0]->cpu_data();
  const Dtype* weight_data = bottom[1]->cpu_data();

  const int count = top[0]->count();
  Dtype* top_data = top[0]->mutable_cpu_data();
  
  vector<int> featmap_shape = bottom[0]->shape();   // including N C H W
  vector<int> weights_shape = bottom[1]->shape();   // N C
  vector<int> top_shape = top[0]->shape();

  const int sample_count = featmap_shape[0];      // number of samples in a batch
  const int featmap_channel = featmap_shape[1];   // channel count of the featmap
  const int featmap_height = featmap_shape[2];    // height of the featmap
  const int featmap_width = featmap_shape[3];     // width of the featmap

  // const int class_count = weights_shape[0];       // class count of the model
  //const int weights_dim = weights_shape[1];       // dimension of the weights, which should be equal to the channel count of the featmap

  // check the equality here
  CHECK(featmap_shape[1] == weights_shape[1]) << "the # of channels in bottom[0] should be equal to the second dimention of bottom[1]";
  /* test
  Dtype sumw = Dtype(0);
  for(int i = 512; i < 2*weights_shape[1]; i++)
  {
    LOG(INFO) << "weight [" << i-512 << "] = " << *(weight_data + i);
    sumw += *(weight_data + i);
  }LOG(INFO) << "sum = " << sumw;*/

  const int feat_size = featmap_channel * featmap_height * featmap_width;
  const int channel_size = featmap_height * featmap_width;

  caffe_set(count, Dtype(0), top_data);
  for (int item_id = 0; item_id < sample_count; ++item_id)
  {
     for (int ch_i = 0; ch_i < featmap_channel; ++ch_i)
     {
       Dtype* feat = (Dtype*)malloc(channel_size * sizeof(Dtype));   // one channel of the featmap 
       Dtype weight = Dtype(0);
       int feat_offset = item_id*feat_size + ch_i * channel_size;
       // int weight_offset = 0; //LOG(INFO) << "*(weight_data + weight_offset) = " << *(weight_data + weight_offset);
       for (int data_i = 0; data_i < channel_size; ++data_i)
       {
         *(feat+data_i) = *(feat_data + feat_offset + data_i);  // copy a channel from feat_data to feat data by data
       }
       weight = *(weight_data + ch_i); 
       
       caffe_axpy(channel_size, weight, feat, top_data+item_id*channel_size);
       free(feat);
       // feat = NULL;
     }
  } 
}

template <typename Dtype>
void WeightedAveLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  
  const Dtype* top_diff = top[0]->cpu_diff();
  // const Dtype* feat_data = bottom[0]->cpu_data();
  const Dtype* weight_data = bottom[1]->cpu_data();

  vector<int> featmap_shape = bottom[0]->shape();   // including N C H W
  // vector<int> weights_shape = bottom[1]->shape();   // N C
  vector<int> top_shape = top[0]->shape();

  const int sample_count = featmap_shape[0];      // number of samples in a batch
  const int featmap_channel = featmap_shape[1];   // channel count of the featmap
  const int featmap_height = featmap_shape[2];    // height of the featmap
  const int featmap_width = featmap_shape[3];     // width of the featmap
  const int feat_size = featmap_channel * featmap_height * featmap_width;
  const int channel_size = featmap_height * featmap_width;

  // const int weights_dim = weights_shape[1];


  if (propagate_down[0])
  {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    for (int item_id = 0; item_id < sample_count; ++item_id)
    {
      for (int ch_i = 0; ch_i < featmap_channel; ++ch_i)        
      {
        const Dtype* item_top_diff = top_diff + item_id*channel_size;
        const Dtype weight = *(weight_data + ch_i); 
        Dtype* item_bottom_diff = bottom_diff + item_id*feat_size + ch_i*channel_size;
        caffe_cpu_scale(channel_size, weight, item_top_diff, item_bottom_diff);
      }
    }
  }

  // if (propagate_down[1])
  // {
  //   Dtype* bottom_diff = bottom[1]->mutable_cpu_diff();   // size = (1, feat_channel)
  //   for (int item_id = 0; item_id < sample_count; ++item_id)
  //   {
  //     for (int ch_i = 0; ch_i < featmap_channel; ++ch_i)        
  //     {
  //       const Dtype* item_top_diff = top_diff + item_id*channel_size;
  //       const Dtype* feat = feat_data + item_id*feat_size + ch_i*channel_size;  // x[i]
  //       Dtype* temp_diff = new Dtype[channel_size];
  //       caffe_mul(channel_size, feat, item_top_diff, temp_diff);  // dJ/dw[i = 1...512] = (dJ/dy) * (dy/dw); dy/dw[i] = x[i];
  //       // summation
  //       Dtype summation = Dtype(0);
  //       for (int i = 0; i < channel_size; ++i)
  //       {
  //         summation += temp_diff[i];
  //       }
  //       Dtype* item_bottom_diff = bottom_diff + item_id*featmap_channel + ch_i;
  //       caffe_set(1, summation, item_bottom_diff);
  //       delete [] temp_diff;
  //     }
  //   }
  // }
}

#ifdef CPU_ONLY
STUB_GPU(WeightedAveLayer);
#endif

INSTANTIATE_CLASS(WeightedAveLayer);
REGISTER_LAYER_CLASS(WeightedAve);

}  // namespace caffe
