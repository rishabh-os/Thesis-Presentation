---
theme: ./theme
class: text-center
highlighter: shiki
lineNumbers: false
drawings:
  persist: false
hideInToc: true
title: >-
  Study of Unsupervised Learning for Images and Videos with Specific
  Applications to CCTV Data
---

# Study of Unsupervised Learning for Images and Videos with Specific Applications to CCTV Data

<div class="cover-divider"/>

<div class="cover-byline">
  Rishabh Wanjari

  20171056
</div>

<!--
Welcome respected panel. Today, I'll be presenting the work I have done in my thesis, titled Study of Unsupervised Learning for Images and Videos with Specific Applications to CCTV Data.
This work was done under the supervision of Aniruddha Pant at AlgoAnalytics, with Sourabh Dube as the expert.
-->

---
---
<Toc />

<!--  I'll be going over the points you see listed here.

I'll set up the importance of the problem, describe the kinds of datasets we're dealing with, and finally go over the models I tried and their results.

So, let's get started.-->
---
layout: image-left
image: https://images.unsplash.com/photo-1515432085503-cabf2fbcd690?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=880&q=80
---

# Why is CCTV monitoring relevant?

<br><br>

CCTV cameras are ubiquitous in today's world. A recent report [1] expected to have more than 1 billion by the end of 2021.

They are used to monitor and surveil a large variety of locations. Parking lots, retailers, roads, warehouses, etc.

<br><br><br><br><br><br><br><br>

[1]: [IHS Markit](https://cdn.ihs.com/www/pdf/IHS-Markit-Technology-Video-surveillance.pdf)

<!--
In modern society, we can find a CCTV camera essentially wherever we go. A recent report (by IHS Markit, a global information provider) expected to have more than 1 billion CCTV cameras in operation by the end of 2021. Being May 2022 now, we have with all likelihood far surpassed that number.

These cameras are used mainly for monitoring and surveillance purposes, mainly in the public sector on streets and in shopping malls, but increasingly so in the private sector as well as home security tools. -->
---

# Problems with CCTV Today

- Most places contain a monitoring room to which all the CCTV footage is sent to. This room is typically only staffed by a single security personnel.
  As a result, it becomes difficult for a single guard to continuously monitor each camera feed.
- At smaller scales, CCTV footage is only reviewed once an incident takes place.

## Proposed Solution

The aim is to develop an **end-to-end** pipeline that monitors these cameras and immeadiately inform the system if any *abnormal* activity occurs.

<Definition title="End-to-end Pipeline" description="An end-to-end pipeline would consist of data ingestion, pre-processing the data, the model itself, and reporting results."/>

<!--
In most places where CCTV monitoring is required, there exists a designated monitoring room where all the CCTV footage is, well, monitored and stored. This room is typically understaffed; in most cases only by a single security guard. This guard, no matter how professional cannot be expected to pay full attention to all the CCTV feeds that may be coming from various parts of a facility. This problem only gets worse with an increase in cameras.

In another case, especially in small deployments, CCTV footage is not even monitored and only seen in case something goes wrong, *after* the incident has occurred.

These issues make it easy for damages to occur. The aim for this work is to develop an end-to-end pipeline that can automatically flag any such incidents and alert the system.

An end-to-end pipeline would consist of data ingestion from camera feeds, pre-processing this video data into separate frames that can be used by a machine learning model, the model itself, and finally the reporting of the outcome of the model.
 -->

---

# Anomalies

Anomalies, especially for video data, can be hard to define concretely. In the real world, the definition of an
anomaly can vary considerably depending on the situation and context.

<Definition title="Anomalies" description="Also known as outliers or novelties, anomalies are defined as data points that deviate significantly
      from the normal data."/>
<div class="w-[400px] mx-auto">

![](/anomaly-example.png)
</div>

<!-- The problem we have at hand is essentially an anomaly detection problem. With video data, anomalies can be hard to define, as they can vary depending on the situation and context.

For example, for a camera monitoring
a store, video footage of burglary, arson and violence have entirely different characteristics. This complexity is why security guards are required to monitor video feeds. We can eliminate this complexity by defining anomalies as data points that deviate significantly from the normal data.

For example, in this graph of sales data, normal prices lie in the range of $25,000 to $75,000. Any price that does not lie in this range is classified as an anomaly.
-->
---

# Supervised or Unsupervised?

Anomalous events, by definition, are an exception rather than the norm. There are an innumerable number of types of anomalies that are possible.

As a result, labeled data is hard to come by. In contrast, unlabeled, normal data is very simple to acquire.

I decided to create a *weakly labeled* data set.

<Definition title="Weakly Labeled Data" description="A dataset that is unlabeled, but certain information about the dataset as a whole is still known."/>

<!--
The nature of anomalies means that there are far more normal data points available than anomalous ones. Additionally, what may be classified as an anomaly in one situation may be considered normal in another. This makes finding labelled data hard to come by.

Due to this, I have decided to use an unsupervised approach. However, since normal data is so much more common that anomalous data, I can create my dataset in such a way that all of my training data consists only of normal data points. This is known as a *weakly labeled* data set. It is weakly labelled because regardless of what each individual frame contains, I can be assured that it only contains normal frames.-->

---

# Datasets

I have included ***eight*** (8) datasets in this work.

| **Dataset**  | **Description**                       | **Anomalous Event(s)**                    | **Dataset Size** |
| ------------ | ------------------------------------- | ----------------------------------------- | ---------------- |
| abbey        | A crossroad with both people and cars | Road accidents                            | 10,726           |
| beach        | An empty beach at night               | People walking around; torches being lit  | 8,404            |
| castro       | A busy street surrounded by buildings | Traffic jams or road accidents            | 4,054            |
| meteor-night | A still sky with no motion            | Appearance of meteors                     | 4,604            |
| motorway     | A busy highway                        | Traffic jams                              | 6,679            |
| tractor      | Tractors passing on a dirt road       | Tractors catching fire or other accidents | 13,504           |
| volcano      | The peak of a volcano                 | The volcano erupting                      | 5,132            |

<!--
In this work, I have covered 8 datasets. Each dataset consists of video clips that have been split into frames. As you can see from the table, these datasets cover a large variety of situations and scenarios. This vareity helps in increasing the architecture's robustness.

I'd like to stress that the goal is to not make a single model that can be fit to all 8 datasets combined, rather, the aim is to make a model architecture that can be trained on each dataset individually and produce satisfactory results on all of them.

I'd like to give you a sense of what some of these datasets contain.-->

---
hideInToc: true
---

# Datasets -- *abbey*

<TwoImages img1="/abbey-normal.jpg" img2="/abbey-anomaly.jpg" desc1="A normal street in England." desc2="The same street after a road accident took place."/>
<!-- First, we have the abbey dataset, in which a camera overlooks a busy street in England. The anomalous event here is a road accident that has taken place. -->
---
hideInToc: true
---

# Datasets -- *motorway*

<TwoImages img1="/motorway-normal.jpg" img2="/motorway-anomaly.jpg" desc1="A countryside highway." desc2="The same highway, now with a traffic jam on one side."/>
<!-- Next, we have the motorway dataset, which is a countryside highway. In this case, the anomaly is a traffic jam that occurs on the left lane. -->
---
hideInToc: true
---

# Datasets -- *otter*

<TwoImages img1="/otter-normal.jpg" img2="/otter-anomaly.jpg" desc1="An otter enclosure at a zoo." desc2="Humans that belong outside the otter enclosure."/>
<!-- Finally, we have the otter dataset, which has a camera monitoring an otter enclosure at a zoo. The anomalous event here is humans enetering the otter enclose, which is anomalous because you would normally find humans in the human enclosure. -->
---
layout: image-right
image: /simple_demo.svg
---

# Methodology

The aim is for the model to learn what a *normal* scene looks like. This is achieved by having the model reconstruct the input images.

As the training data is only normal data, the model will excel at reconstructing normal data.

For data that does not fit this normal data, the model will be unable to reconstruct the image properly.

<!--
As illustrated by the figure on the right, if normal data is represented with a `0` and anomalous data with `1`, we train a model to faithfully reconstruct the normal data. When the model encouters a data point that is not normal, it will fail to provide a faithful reconstruction. We can use this to discern between anomalies and normality, by setting a threshold for the error in the reconstruction.

Now, we need to address the problem of comparing two images and getting a reconstruction error. -->
---

# Comparing Images

The most common way of comparing two images is to use MSE, given by
$$
MSE(x,y)=\frac1N\sum^N_{i=1}(x_i-y_i)^2
$$

An alternative is to use the Structural Similarity Index (SSIM):
$$
SSIM(x,y)=\frac{(2\mu_x\mu_y+C_1)(2\sigma_{xy}+C_2)}{(\mu_x^2+\mu_y^2+C_1)(\sigma_x^2+\sigma_y^2+C_2)}
$$
SSIM compares the images in patches, taking into account the spatial distribution, luminance and contrast of the image.

- Mean $\rightarrow$ Luminance
- Variance $\rightarrow$ Contrast
- Covariance $\rightarrow$ Similarity

<!--
However, MSE has it's limitations. It is unbounded, doesn't take into account the structure of the image, and importantly, images with low MSE are not guaranteed to look even remotely similar, as we will see soon.

SSIM is based on the model of human perception. It measures the similarity of the three aspects: luminance,  contrast,  and similarity using the mean, variance and covariance respectively. It does so in patches across the entire image. The final SSIM is obtained by computing the average SSIM of all the patches in the image. Once you do all the calculations involved, you arrive at this formula.
-->

---
hideInToc: true
---

# Hypersphere

<div class="w-[30rem] mx-auto">

![](/mse-hypersphere.png)
</div>

[Wang, Z. et al: Image quality assessment: From error visibility to structural similarity (2004)](https://ieeexplore.ieee.org/document/1284395)

<!--
This figure serves as an excellent demonstration of why MSE is not suitable for image data. What you can see here is that all the images on the rim have exactly the same MSE.

However, it is very obvious that all of these images are not the same. The image that is visually most similar to the initial image *just so happens to be* the one with the best SSIM.

Now we have a way to compare two images and quantify their similarity. We now need to get around to actually constructing the second image that will be compared to the original. This is where the machine learning model comes in.-->

---

# Convolutions

A convolution is a process of taking a matrix (of size smaller than the input image) called the kernel and passing it over an input image to transform the image based on the kernel.

$$
G[m,n]=(f*h)[m,n]=\sum_j\sum_kh[j,k]f[m-j,n-k]
$$

<div class="w-[40rem] mx-auto">

![](/cnn_best.gif)
</div>

<!--
I have used a variety of ML models in this work. What they all have in common is that they all use convolutional layers. A convolutional layer performs the convolution operation on an image and returns the resulting image.

In machine learning terms, a convolution is the process of taking an image, and multiplying it in steps with a weight matrix *- the kernel -* and computing the final output.

In mathematical form, it looks like this.

The figure is the same operation in visual form. We take a 5x5 figure and convolve it with a 3x3 kernel to get a 3x3 figure.

Now we can move onto talking about the ML architectures themselves.
 -->
---

# Autoencoders

They are capable of learning dense representations of the data.

An autoencoder (AE) is trained to replicate its input. To prevent it from learning the identity mapping, we introduce additional constraints in the network.

<div class="w-[20rem] mx-auto">

![](/sandwich.png)
</div>

                          Input           Dense            Output
                                      Representation
<!-- Hack for image label at the bottom lol -->

<!--
The first of these is the autoencoder.
You can think of autoencoders as the machine learning equivalent of compressing and uncompressing an image. In keeping with this analogy, they are trained to get as close to lossless compression as possible.

The way we constrain this AE is by limiting the number of nodes in the hidden layers. This set up specifically is referred to as an undercomplete autoencoder.


An autoencoder is symmetrical, and consists of two parts: the encoder on the left (compressor) and the decoder (decompressor) on the right. The purple part in the middle is the dense representation of the input that the autoencoder has learnt, in order words, the compressed version of the data.
-->

---

# Variational Autoencoders

VAEs fall into the class of generative models. They are probabilistic in nature.

With AEs there is no guarantee that any point from the latent space will generate a new data point.

Each point in the penultimate layer of the encoder is mapped to a Gaussian distribution with a mean $\mu$ and variance $\sigma$.


<div class="w-[30rem] mx-auto">

![](/vae.png)
</div>

<!--
Up next, we have a variation on the AE architecture called, well, the variational autoencoder. This is a generative model.

With a normal AE architecture, there is no guarantee that if we randomly sample a point in the latent space and pass it through the decoder, we will end up with a data point that is part of the input data set. VAEs are a remedy to this problem.

They follow the same general architecture as that of autoencoders, with some modifications to the latent space representations. Instead of mapping to a fixed point, each point in the penultimate layer of the encoder is mapped to a Gaussian distribution. The decoder then randomly samples from this distribution to generate an image.
-->
---

# GANs

Simple idea: have two neural networks compete against each other. Tediously difficult in implementation.

The two networks are called the generator $G$ and the discriminator $D$.
<div class="w-[30rem] mx-auto">

![](/gan.png)
</div>

<!-- Problems faced : oscillation of parameters, mode collapse unbalanced generator and discriminator. -->
Problems faced:
- Oscillation of parameters
- Mode collapse
- Unbalanced networks

[Hayes J. et al. LOGAN: Evaluating Privacy Leakage of Generative Models Using Generative Adversarial Networks (2017)](https://arxiv.org/abs/1705.07663)
<!-- We now take a small detour to explain GANs so that I can explain the upcoming model better.

The idea behind GANs is simple: have two networks compete against each other and hope that, much like in the real world, the competition makes them excel. However, it is difficult to implement in practice.

A GAN consists of two networks: a generator, whose job is to generate new images from a random sample, and a discriminator, whose job is to tell if the generated image is real or fake. You can think of the generator as the decoder part of the VAE we just discussed.

Oscillations of parameters: the model parameters never converge Since both the generator and dis-
criminator compete against each other, their parameters may continue to oscillate and never stabilise.
Many factors can contribute to this behaviour, making GANs very sensitive to initial parameters.
• Mode collapse: the generator’s outputs become less diverse This can happen if the generator gets good
at producing a particular kind of image that can fool the discriminator. Since we reward this, it will
continue to produce only those particular images and will not produce other kinds of images. Mode
collapse is rarely to a single point, but partial collapses are common.
• Unbalanced generator and discriminator: Both networks can be unbalanced in the amount of infor-
mation each can process. As an extreme example, consider a dense, fully connected network as the
discriminator and a convolutional network as the generator. The discriminator is not powerful enough
to provide good feedback data for the generator, and as a result, the generator will not learn to generate
realistic images. However, currently, there is no metric to compare the power of two networks.-->
---

# Bidirectional GANs

Traditional GANs are structured learn the mappings from a simple distribution to a complex dataset.

BiGANs provide a way to learn the inverse mapping and, more importantly, the ability to retrieve a reconstruction error from a GAN based model.

<div class="w-[40rem] mx-auto">

![](/bigan.png)
</div>

<br>

[Donahue J. et al: Adversarial feature learning (2016).](https://doi.org/10.48550/arXiv.1605.09782)
<!--
As we saw just now, GANs take a random noise vector and map it to a complex dataset. However, what we want it to map from complex dataset to complex dataset, so that we have two images to compare two. This is where BiGANs come in. We introduce an encoder that is able to map from the training data to the smaller vector space.
I'll leave out the exact details of what this does, but the important part is that, now, using BiGANs, we are able to provide an image to the BiGAN network and receive a reconstructed image as the output.
-->

---

# Results

| **Dataset**              | **Precision** | **Recall** | **F1**   | **AUROC** | **AUPRC** |
| ------------------------ | ------------- | ---------- | -------- | --------- | --------- |
| Previous Model (Average) | 0.77          | 0.76       | 0.77     | 0.68      | 0.76      |
| BiGAN (Average)          | 0.7           | 0.64       | 0.63     | 0.71      | 0.72      |
| VAE (Average)            | 0.86          | 0.82       | 0.83     | 0.85      | 0.87      |
| AE (Average)             | **0.88**      | **0.94**   | **0.91** | **0.92**  | **0.91**  |

<!--
I undertook this project because the previous model's performance was not good enough, and was in dire need of improvement. I tried the three different models I have just described: the AE, the VAE and the BiGAN.

The BiGAN model performed below expectations, as being the most complex I expected it to perform the best. However, it failed to do so. We attribute this to the traditional difficulties encountered in training GANs, although we tried our best to mitigate them.

However, the AE based models performed much better. With these results, I can safely say that I have achieved the goal of improving model performance with the AE based model.

Seeing as the AE model performs the best in the metrics we have chosen, I will now focus on the results from this model specifically.-->

---
hideInToc: true
---

# Performance

<div class="w-[40rem] mx-auto">

![](/ae_perf.png)
</div>

<!--
An important focus of this work was to ensure that the model is performant. To that end, I have also achieved good results.

This is a performance graph of the AE based model on two systems: a Colab GPU that is accessible to everyone, and on a CPU of a commonly available laptop.

While training on a GPU clearly has it's advantages, allowing model training to be completed in half an hour on average, we can see that even training on a CPU is still feasible, taking slightly longer at an average of 2 hours. -->
---
hideInToc: true
---

# Demonstration
<div grid="~ cols-2 gap-2">
<div class="my-auto mx-auto">
  <LocalVideo filepath="/video.mp4"></LocalVideo>
</div>

<div class="my-auto mx-auto">

![Frame data](/frames_tractor.png)

</div>
</div>

<!--  Here is a video demonstration of the model's results on the tractor dataset. On the left we have the test video clip, and on the right is the corresponding graph for the frame level reconstruction error. The red line represents the reconstruction threshold and the yellow highlighted region represents the truely anomalous data. Here, the anomalous event is one of the tractors malfunctioning, resulting in a large plume of smoke.

What is important to notice is that the blue line, which represents the reconstruction error of each frame, goes up just as the anomaly occurs and comes back down once it is over. This means that the model is very precise at the task of anomaly detection. -->
---
hideInToc: true
clicks: 5
---

# Summary

<v-clicks>

- Set up the problem, and explored the type of data that was used.
- Explored the various ways of computing the reconstruction error for two images.
- Explored three different unsupervised model architectures to attempt to solve this issue.
- Successfully found a model with optimal parameters, metrics and performance that is ready to be deployed.

</v-clicks>

<v-clicks at="5">
<div>
<h2 class="mt-10 mb-4">Takeaway</h2>
This model has the potential to increase the effectiveness of security and surveillance systems by lowering the barrier of entry and making such systems accessible to everyone.
</div>

</v-clicks>

<!--
With all this said, I'd like to quickly go over everything that I have covered today.
- To begin with, I set the scene with CCTV cameras and explained their current shortcomings. I also introduced the various kinds of data sets I have worked with.
- I described methods of computing the reconstruction error between two images, choosing SSIM as the better method.
- I explored three different unsupervised model architectures. From these three, I found that the autoencoder model works the best, with great metric scores and performance.

Finally, the key point I would like you to take away from my work is that I have developed an end-to-end pipeline that is ready to be deployed. The performance graphs I showed earlier let us deploy this model essentially anywhere, which makes this technology more accessible to everyone.
-->

---
hideInToc: true
---

<div class="fin">
Fin
</div>

<!--
And with that, we come to and end of my presentation. Thank you.
-->
