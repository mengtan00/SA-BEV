<div align="center">
<h1>SA-BEV</h1>
<h3>[ICCV2023] SA-BEV: Generating Semantic-Aware Bird's-Eye-View Feature for Multi-view 3D Object Detection</h3>
</div>



<div align="center">
  <img src="resources/sabev.png" width="800"/>
</div><br/>

## News

- **2023.07.14** SA-BEV is accepted by ICCV 2023.

## Main Results

| Config                                                                    | mAP        | NDS        | 
| ------------------------------------------------------------------------- | ---------- | ---------- |
| [**SA-BEV-R50**](configs/sabev/sabev-r50.py)                            | 35.5       | 46.7       | 
| [**SA-BEV-R50-MSCT**](configs/sabev/sabev-r50-msct.py)                  | 37.0       | 48.8       |
| [**SA-BEV-R50-MSCT-CBGS**](configs/sabev/sabev-r50-msct-cbgs.py)        | 38.7       | 51.2       | 
The latency includes Network/Post-Processing/Total.


## Get Started

#### Installation and Data Preparation

1. Please refer to [getting_started.md](docs/en/getting_started.md) for installing BEVDet as mmdetection3d. 
2. Prepare nuScenes dataset as introduced in [nuscenes_det.md](docs/en/datasets/nuscenes_det.md) and create the pkl for SA-BEV by running:

```shell
python tools/create_data_bevdet.py
```

## Acknowledgement

This project is not possible without multiple great open-sourced code bases. We list some notable examples below.

- [BEVDet](https://github.com/HuangJunJie2017/BEVDet)
- [BEVDepth](https://github.com/Megvii-BaseDetection/BEVDepth)
- [open-mmlab](https://github.com/open-mmlab)
- [CenterPoint](https://github.com/tianweiy/CenterPoint)
- [Lift-Splat-Shoot](https://github.com/nv-tlabs/lift-splat-shoot)

## Bibtex

If SA-BEV is helpful for your research, please consider citing the following BibTeX entry.

