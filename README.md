**Welcome!**

Visualizations discussed during our meetings can be found in the `output/display` folder.

Link to my UW-Madison Box with one file's worth of data in it: https://uwmadison.box.com/s/ah0lfjjc92ocsp4rtyr0o595q24bsucc

Link to the USGS data used as ground truth for seismic events https:

https://earthquake.usgs.gov/earthquakes/map/?currentFeatureId=ak0225r281m5&extent=59.74533,-151.56189&extent=62.1655,-142.77283&range=search&sort=oldest&listOnlyShown=true&timeZone=utc&search=%7B%22name%22:%22Search%20Results%22,%22params%22:%7B%22starttime%22:%222022-05-04%2000:00:00%22,%22endtime%22:%222022-05-24%2023:59:59%22,%22maxlatitude%22:65.422,%22minlatitude%22:52.133,%22maxlongitude%22:-132.188,%22minlongitude%22:-167.344,%22minmagnitude%22:2.5,%22orderby%22:%22time%22%7D%7D

The events I have primarily focused occur on 5/5/2022 and 5/19/2022.

**Program descriptions:**

- **FK.py:**  
  Performs a frequency-wavenumber transform of all channels over a specified time frame. Parameters in code are starting time, duration (in seconds), and frequency range of interest.

- **STFT.py:**  
  Performs a short-time Fourier transform of all channels over a specified timeframe and channel set. Parameters in code are starting time, duration (in seconds), channel range, and frequency range of interest. Additionally, `seg_length` and `nperseg` can be modified for increased or decreased granularity.

- **PSD_plotter.py:**  
  Very similar to `STFT.py`. I would recommend just using that.

- **heatmap_zoom_in.py:**  
  Plots the base DAS data (strain measurements) as a heatmap of all channels over a specified time.

- **wavelet_plotter.py:**  
  Plots HAAR wavelets for all channels over a specified time period.
