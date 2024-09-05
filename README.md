**Welcome!**

Visualizations discussed during our meetings can be found in the `output/display` folder.

**Program descriptions:**

- **FK.py:**  
  Performs a frequency-wavenumber transform of all channels over a specified time frame. Parameters in code are starting time, duration (in seconds), and frequency range of interest.

- **STFT.py:**  
  Performs a short-time Fourier transform of all channels over a specified timeframe. Parameters in code are starting time, duration (in seconds), and frequency range of interest. Additionally, `seg_length` and `nperseg` can be modified for increased or decreased granularity.

- **PSD_plotter.py:**  
  Very similar to `STFT.py`. I would recommend just using that.

- **heatmap_zoom_in.py:**  
  Plots the base DAS data (strain measurements) as a heatmap of all channels over a specified time.

- **wavelet_plotter.py:**  
  Plots HAAR wavelets for all channels over a specified time period.
