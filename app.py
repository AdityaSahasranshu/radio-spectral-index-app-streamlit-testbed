# -*- coding: utf-8 -*-
"""
Created on Tue Dec 30 18:53:30 2025

@author: adity
"""

import streamlit as st
import os
import numpy as np
import warnings
import tempfile
from astropy.io import fits
from astropy import units as u
from astropy.wcs import WCS
from astropy.stats import mad_std
from radio_beam import Beam
from reproject import reproject_interp
import matplotlib.pyplot as plt

# --- 1. SETUP ---
st.set_page_config(page_title="Spectral Index Pipeline", layout="wide")
warnings.filterwarnings('ignore')
st.title("Spectral Index Pipeline")

# --- 2. THE CLASS (Modified ONLY for Web I/O) ---
class RadioMap:
    def __init__(self, uploaded_file, name="Map"):
        self.name = name
        
        # WEB CHANGE: Handle uploaded bytes instead of file path
        self.tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".fits")
        self.tfile.write(uploaded_file.getvalue())
        self.tfile.close()
        self.filepath = self.tfile.name
        
        self.data = None
        self.header = None
        self.wcs = None
        self.beam = None
        self.freq = None
        
        self.load_data()

    def load_data(self):
        # WEB CHANGE: No need to check os.path.exists (tempfile guarantees it)
        
        with fits.open(self.filepath) as hdul:
            hdu = hdul[0] if len(hdul) > 0 and hdul[0].data is not None else hdul[1]
            self.header = hdu.header
            self.data = np.squeeze(hdu.data)
            self.wcs = WCS(self.header).celestial
            
        st.write(f"**Loaded {self.name}**") # WEB CHANGE: print -> st.write
        self._get_frequency()
        self._get_beam()
        self._check_units()

    def _get_frequency(self):
        keys = ['RESTFRQ', 'FREQ', 'CRVAL3'] 
        found = False
        for k in keys:
            if k in self.header:
                val = self.header[k]
                if val > 1e6:
                    self.freq = val * u.Hz
                    st.info(f"Freq {self.name}: {self.freq.to(u.MHz):.2f}") # WEB CHANGE: print -> st.info
                    found = True
                    break
        if not found:
            # WEB CHANGE: input() -> st.number_input() with st.stop()
            st.warning(f"[!] Frequency missing for {self.name}")
            val = st.number_input(f"Enter Frequency for {self.name} (MHz):", min_value=0.0, step=1.0, key=f"freq_{self.name}")
            
            if val == 0.0:
                st.error("Please enter a frequency > 0 to continue.")
                st.stop() # PAUSE HERE until user inputs value
            
            self.freq = val * u.MHz

    def _get_beam(self):
        try:
            self.beam = Beam.from_fits_header(self.header)
            st.info(f"Beam {self.name}: {self.beam}") # WEB CHANGE: print -> st.info
        except:
            # WEB CHANGE: input() -> st.number_input()
            st.warning(f"[!] Beam info missing in {self.name}.")
            
            c1, c2, c3 = st.columns(3)
            bmaj = c1.number_input(f"{self.name} Major Axis (arcsec):", min_value=0.0, key=f"bmaj_{self.name}")
            bmin = c2.number_input(f"{self.name} Minor Axis (arcsec):", min_value=0.0, key=f"bmin_{self.name}")
            bpa = c3.number_input(f"{self.name} PA (deg):", value=0.0, key=f"bpa_{self.name}")
            
            if bmaj == 0.0 or bmin == 0.0:
                st.error("Please enter Beam Major/Minor axis to continue.")
                st.stop() # PAUSE HERE

            self.beam = Beam(major=bmaj*u.arcsec, minor=bmin*u.arcsec, pa=bpa*u.deg)

    def _check_units(self):
        """Detects mJy vs Jy and converts to Jy if needed."""
        unit_str = self.header.get('BUNIT', '').lower()
        
        if 'mjy' in unit_str:
            st.write(f"[{self.name}] Converting mJy -> Jy")
            self.data = self.data / 1000.0
            self.header['BUNIT'] = 'Jy/beam'
        elif 'jy' in unit_str:
            pass 
        else:
            # WEB CHANGE: input() -> st.radio()
            st.warning(f"[!] Unit unknown for {self.name}. Is this map in mJy/beam?")
            choice = st.radio(f"Select units for {self.name}:", ("Select...", "Yes (mJy)", "No (Jy)"), key=f"unit_{self.name}")
            
            if choice == "Select...":
                st.stop() # PAUSE HERE
            
            if choice == "Yes (mJy)":
                self.data = self.data / 1000.0

    def cleanup(self):
        if os.path.exists(self.filepath):
            os.unlink(self.filepath)


# --- 3. CONVOLUTION FUNCTION (Logic Unchanged) ---
def convolve_to_common(source_map, target_beam):
    if source_map.beam == target_beam:
        st.write(f"{source_map.name} already matches target beam.")
        return source_map.data

    st.write(f"--- Convolving {source_map.name} ---")
    st.write(f"From: {source_map.beam}")
    st.write(f"To:   {target_beam}")
    
    try:
        # 1. Calculate Kernel
        kernel = target_beam.deconvolve(source_map.beam)
        kernel_pix = kernel.as_kernel(source_map.wcs.proj_plane_pixel_area()**0.5)
        
        # 2. Convolve (FFT)
        from astropy.convolution import convolve_fft
        convolved_data = convolve_fft(source_map.data, kernel_pix, allow_huge=True)
        
        # 3. Apply Beam Area Scaling
        source_area = source_map.beam.sr.value
        target_area = target_beam.sr.value
        scale_factor = target_area / source_area
        
        st.write(f"[Physics] Scaling Factor: {scale_factor:.4f}")
        return convolved_data * scale_factor
        
    except ValueError:
        st.error("[CRITICAL ERROR] Deconvolution failed despite Common Beam logic.")
        return source_map.data


# --- 4. MAIN WORKFLOW ---
def calculate_spectral_index_workflow():
    
    # 1. INPUTS (File Uploader)
    st.sidebar.header("Upload Files")
    f1 = st.sidebar.file_uploader("Upload FITS File 1", type=["fits"])
    f2 = st.sidebar.file_uploader("Upload FITS File 2", type=["fits"])

    if not f1 or not f2:
        st.info("Waiting for both FITS files to be uploaded...")
        st.stop()

    # Create Objects (This triggers the checks inside the Class)
    # If info is missing, the Class will STOP the script here until user fills it.
    map1 = RadioMap(f1, "Map 1")
    map2 = RadioMap(f2, "Map 2")

    st.markdown("---")
    st.subheader("Processing...")

    # 2. DEFINE COMMON BEAM
    max_axis_1 = max(map1.beam.major, map1.beam.minor)
    max_axis_2 = max(map2.beam.major, map2.beam.minor)
    common_size = max(max_axis_1, max_axis_2) * 1.01 
    common_beam = Beam(major=common_size, minor=common_size, pa=0*u.deg)
    
    st.success(f"Common Beam Defined: {common_beam.major.to(u.arcsec):.2f}")

    # 3. CONVOLVE
    with st.spinner("Convolving Maps..."):
        data1_conv = convolve_to_common(map1, common_beam)
        data2_conv = convolve_to_common(map2, common_beam)

    # 4. REGRIDDING
    st.write("--- Regridding Map 1 to Map 2 Grid ---")
    with st.spinner("Regridding..."):
        data1_aligned, footprint = reproject_interp(
            # --- 🔍 PASTE THIS DIAGNOSTIC BLOCK HERE ---
            st.write("--- 🔍 DIAGNOSTICS (LOFAR vs VLASS) ---")
            
            # 1. Check if the maps are just empty space (NaNs)
            # If 'Percent Empty' is 100%, your maps do not overlap in the sky.
            nan_1 = np.isnan(d1_aligned).sum()
            total_pix = d1_aligned.size
            st.write(f"**Map 1 (Aligned) Empty Pixels:** {nan_1}/{total_pix} ({nan_1/total_pix:.1%})")
            
            # 2. Check the Max Flux
            # LOFAR and VLASS are both bright. If these numbers are tiny (like 1e-10), math failed.
            max_1 = np.nanmax(d1_aligned)
            max_2 = np.nanmax(d2_conv)
            st.write(f"**Map 1 Max Flux:** {max_1:.5e} Jy/beam")
            st.write(f"**Map 2 Max Flux:** {max_2:.5e} Jy/beam")
            
            if nan_1 == total_pix:
                st.error("🚨 CRITICAL ERROR: Map 1 is 100% empty after regridding. The two FITS files do not cover the same patch of sky.")
                st.stop()
            
            # 3. Quick visual check of the raw data (ignores headers)
            # This proves if data exists in the array at all
            st.write("Quick look at raw array data (Map 1 vs Map 2):")
            cols = st.columns(2)
            cols[0].image((d1_aligned - np.nanmin(d1_aligned)) / (np.nanmax(d1_aligned) - np.nanmin(d1_aligned)), caption="Map 1 Aligned", clamp=True)
            cols[1].image((d2_conv - np.nanmin(d2_conv)) / (np.nanmax(d2_conv) - np.nanmin(d2_conv)), caption="Map 2 Convolved", clamp=True)
        # --- END DIAGNOSTICS ---
            (data1_conv, map1.wcs),
            map2.wcs,
            shape_out=map2.data.shape,
            order=3
        )

    # 5. NOISE & MASKING
    rms_1 = mad_std(data1_aligned, ignore_nan=True)
    rms_2 = mad_std(data2_conv, ignore_nan=True)
    st.write(f"Map 1 RMS: {rms_1:.4e} | Map 2 RMS: {rms_2:.4e}")

    # WEB CHANGE: input() -> st.number_input
    sigma_thresh = st.number_input("Enter Sigma Threshold (e.g., 3.0):", min_value=0.0, value=3.0, step=0.5)
    
    # Button to finalize calculation (prevents constant re-calc when changing sigma)
    # Checkbox for debugging (Add this before the button)
    use_mask = st.checkbox("Apply 3-Sigma Masking?", value=True)

    if st.button("Calculate Final Map"):
        # Define frequencies (needed for both modes)
        v1 = map1.freq.to(u.Hz).value
        v2 = map2.freq.to(u.Hz).value
        
        if use_mask:
            # --- STANDARD MODE (STRICT MASKING) ---
            mask = (data1_aligned > sigma_thresh*rms_1) & (data2_conv > sigma_thresh*rms_2)
            
            # [FIX]: These lines were missing! We must define S1/S2 here.
            S1 = data1_aligned[mask]
            S2 = data2_conv[mask]
            
            alpha_map = np.full_like(map2.data, np.nan)
            
            with np.errstate(invalid='ignore', divide='ignore'):
                # Now S1 and S2 exist, so this math works
                alpha_vals = np.log10(S1 / S2) / np.log10(v1 / v2)
                alpha_map[mask] = alpha_vals
                
        else:
            # --- DEBUG MODE (NO MASKING) ---
            st.warning("⚠️ Masking disabled! Output will show raw math (noisy).")
            
            # We use the FULL arrays here, not S1/S2
            # Added a tiny number (1e-9) to prevent division by zero errors
            with np.errstate(invalid='ignore', divide='ignore'):
                alpha_map = np.log10(data1_aligned / (data2_conv + 1e-9)) / np.log10(v1 / v2)

        # --- DISPLAY & DOWNLOAD (Common to both) ---
        fig, ax = plt.subplots(figsize=(10,5))
        im = ax.imshow(alpha_map, origin='lower', cmap='jet', vmin=-2.0, vmax=0.5)
        plt.colorbar(im, label="Spectral Index")
        st.pyplot(fig)
        
        # Download
        out_header = map2.header.copy()
        out_header['HISTORY'] = 'Spectral Index Map'
        # Save to temp path
        output_path = "/tmp/alpha.fits"
        fits.writeto(output_path, alpha_map, out_header, overwrite=True)
        
        with open(output_path, "rb") as f:
            st.download_button("Download FITS", f, "spectral_index.fits")
        
        # 6. CALCULATION
        st.write("--- Calculating Alpha ---")
        S1 = data1_aligned[mask]
        S2 = data2_conv[mask]
        v1 = map1.freq.to(u.Hz).value
        v2 = map2.freq.to(u.Hz).value
        
        alpha_map = np.full_like(map2.data, np.nan)
        
        with np.errstate(invalid='ignore', divide='ignore'):
            alpha_vals = np.log10(S1 / S2) / np.log10(v1 / v2)
            alpha_map[mask] = alpha_vals

        # 7. SAVE & DOWNLOAD
        out_header = map2.header.copy()
        try:
            out_header.update(common_beam.to_header_keywords())
            out_header['HISTORY'] = f'Convolved to common beam: {common_size.to(u.arcsec):.2f}'
        except: pass
        
        # Save to temp path for download
        output_path = "/tmp/spidx_common.fits"
        fits.writeto(output_path, alpha_map, out_header, overwrite=True)
        
        st.success("Calculation Complete!")
        
        # Quick Preview
        fig, ax = plt.subplots(figsize=(10,4))
        im = ax.imshow(alpha_map, origin='lower', cmap='jet', vmin=-2, vmax=0.5)
        plt.colorbar(im, label="Spectral Index")
        st.pyplot(fig)
        
        # Download Button
        with open(output_path, "rb") as f:
            st.download_button("Download Result FITS", f, "spidx_common.fits", "image/fits")

    # Cleanup temp files
    map1.cleanup()
    map2.cleanup()

if __name__ == "__main__":
    calculate_spectral_index_workflow()