<p align="center">

  <h1 align="center"><a href="https://sairajk.github.io/easi-tex" target="_blank">EASI-Tex: Edge-Aware Mesh Texturing from Single Image</a></h1>

  <p align="center">
    <a href="https://sairajk.github.io/" target="_blank"><strong>Sai Raj Kishore Perla</strong></a>
    ·
    <a href="https://yizhiwang96.github.io/" target="_blank"><strong>Yizhi Wang</strong></a>
    ·
    <a href="https://www.sfu.ca/~amahdavi/" target="_blank"><strong>Ali Mahdavi-Amiri</strong></a>
    ·
    <a href="https://www.cs.sfu.ca/~haoz/" target="_blank"><strong>Hao (Richard) Zhang</strong></a>
    <br />
    <i>ACM Transactions on Graphics (Proceedings of SIGGRAPH), 2024</i>    
  </p>

  <p align="center">
    <a href="" target="_blank"><strong>arXiv</strong></a>
    |
    <a href="https://sairajk.github.io/easi-tex" target="_blank"><strong>Project Page</strong></a>
  </p>

  <div  align="center">
    <a>
      <img src="./static/assets/v2_teaser.jpg" alt="Logo" width="100%">
    </a>
  </div>
</p>

We present a novel approach for single-image mesh texturing, which employs a diffusion model with judicious conditioning to seamlessly transfer an object's texture from a single RGB image to a given 3D mesh object. We do not assume that the two objects belong to the same category, and even if they do, there can be significant discrepancies in their geometry and part proportions. Our method aims to rectify the discrepancies by conditioning a pre-trained Stable Diffusion generator with edges describing the mesh through ControlNet, and features extracted from the input image using IP-Adapter to generate textures that respect the underlying geometry of the mesh and the input texture without any optimization or training. We also introduce Image Inversion, a novel technique to quickly personalize the diffusion model for a single concept using a single image, for cases where the pre-trained IP-Adapter falls short in capturing all the details from the input image faithfully. Experimental results demonstrate the efficiency and effectiveness of our edge-aware single-image mesh texturing approach, coined EASI-Tex, in preserving the details of the input texture on diverse 3D objects, while respecting their geometry.

## Citation

If you found our work helpful, please consider citing:

```bibtex
@article{perla2024easitex,
    title={EASI-Tex: Edge-Aware Mesh Texturing from Single Image},
    author = {Perla, Sai Raj Kishore and Wang, Yizhi and Mahdavi-Amiri, Ali and Zhang, Hao},
    journal = {ACM Transactions on Graphics (TOG)},
    publisher = {ACM New York, NY, USA},
    year = {2024},
    volume = {43},
    number = {4},
    articleno = {40},
    doi = {10.1145/3658222},
    url = {https://github.com/sairajk/easi-tex},
}
```
