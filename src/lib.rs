use anyhow::{Context, Result};
use opencv::core::CV_32F;
use opencv::{
	core::{self, BorderTypes, CV_8UC1, CV_64F, Mat, Scalar},
	imgcodecs, imgproc,
	prelude::*,
};
use std::path::Path;

const RESIZE_DIM: i32 = 100;
const BLUR_SIGMA: f64 = 15.0;

#[derive(Debug)]
pub struct GradientResult {
	pub start_color: String,
	pub end_color: String,
	pub angle: f64,
}

pub fn extract_gradient_hex(image_path: &Path) -> Result<GradientResult> {
	let img = imgcodecs::imread(
		image_path.to_str().context("Not a valid filepath")?,
		imgcodecs::IMREAD_COLOR,
	)
	.context("Failed to read image")?;

	if img.empty() {
		anyhow::bail!("Image is empty at {:?}", image_path);
	}

	let size = img.size()?;
	let mut small = Mat::default();
	imgproc::resize(
		&img,
		&mut small,
		core::Size::new(RESIZE_DIM, RESIZE_DIM * size.height / size.width),
		0.0,
		0.0,
		imgproc::INTER_AREA,
	)?;

	let mut blurred = Mat::default();
	imgproc::gaussian_blur(
		&small,
		&mut blurred,
		core::Size::new(0, 0),
		BLUR_SIGMA,
		BLUR_SIGMA,
		BorderTypes::BORDER_REFLECT as i32,
		core::AlgorithmHint::ALGO_HINT_ACCURATE,
	)?;

	let mut gray = Mat::default();
	imgproc::cvt_color(
		&blurred,
		&mut gray,
		imgproc::COLOR_BGR2GRAY,
		0,
		core::AlgorithmHint::ALGO_HINT_DEFAULT,
	)?;

	let mut grad_x = Mat::default();
	imgproc::sobel(
		&gray,
		&mut grad_x,
		CV_64F,
		1,
		0,
		5,
		1.0,
		0.0,
		BorderTypes::BORDER_CONSTANT as i32,
	)?;

	let mut grad_y = Mat::default();
	imgproc::sobel(
		&gray,
		&mut grad_y,
		CV_64F,
		0,
		1,
		5,
		1.0,
		0.0,
		BorderTypes::BORDER_CONSTANT as i32,
	)?;

	let mut mag = Mat::default();
	let mut angle_rad = Mat::default();
	core::cart_to_polar(&grad_x, &grad_y, &mut mag, &mut angle_rad, false)?;

	let mut max_val = 0.0;
	core::min_max_loc(
		&mag,
		None,
		Some(&mut max_val),
		None,
		None,
		&core::no_array(),
	)?;
	let threshold = 0.1 * max_val;

	let mut valid_mask = Mat::default();
	imgproc::threshold(
		&mag,
		&mut valid_mask,
		threshold,
		255.0,
		imgproc::THRESH_BINARY,
	)?;
	let mut valid_mask_output = valid_mask.clone();
	valid_mask.convert_to(&mut valid_mask_output, CV_8UC1, 1.0, 0.0)?;

	let non_zero_count = core::count_non_zero(&valid_mask_output)?;
	let dominant_angle = if non_zero_count < 10 {
		0.0
	} else {
		let mask_data = valid_mask_output.data_typed::<u8>()?;
		let angle_data = angle_rad.data_typed::<f64>()?;
		let cols = angle_rad.cols() as usize;
		let rows = angle_rad.rows() as usize;

		let mut sum_cos = 0.0;
		let mut sum_sin = 0.0;
		let mut count = 0;

		for y in 0..rows {
			for x in 0..cols {
				let idx = y * cols + x;
				if mask_data[idx] != 0 {
					let a = angle_data[idx];
					let double_angle = 2.0 * a;
					sum_cos += double_angle.cos();
					sum_sin += double_angle.sin();
					count += 1;
				}
			}
		}

		if count == 0 {
			0.0
		} else {
			let avg_cos = sum_cos / count as f64;
			let avg_sin = sum_sin / count as f64;
			0.5 * avg_sin.atan2(avg_cos)
		}
	};

	let dx = dominant_angle.cos();
	let dy = dominant_angle.sin();
	let cartesian_angle_rad = f64::atan2(-dy, dx);
	let angle = (90.0 - cartesian_angle_rad.to_degrees()).rem_euclid(360.0);

	let h = blurred.rows();
	let w = blurred.cols();

	let mut t = Mat::new_rows_cols_with_default(h, w, CV_32F, Scalar::all(0.0))?;
	{
		let cols = t.cols() as usize;
		let t_data = t.data_typed_mut::<f32>()?;

		for y in 0..h {
			for x in 0..w {
				let idx = (y as usize) * cols + (x as usize);
				t_data[idx] = (x as f32) * dx as f32 + (y as f32) * dy as f32;
			}
		}
	}

	let mut min_val = 0.0;
	let mut max_val = 0.0;
	core::min_max_loc(
		&t,
		Some(&mut min_val),
		Some(&mut max_val),
		None,
		None,
		&core::no_array(),
	)?;

	let threshold_low = min_val + 0.15 * (max_val - min_val);
	let threshold_high = max_val - 0.15 * (max_val - min_val);

	let mut start_mask = Mat::default();
	core::in_range(
		&t,
		&Scalar::all(f64::NEG_INFINITY),
		&Scalar::all(threshold_low as f64),
		&mut start_mask,
	)?;

	let mut end_mask = Mat::default();
	core::in_range(
		&t,
		&Scalar::all(threshold_high as f64),
		&Scalar::all(f64::INFINITY),
		&mut end_mask,
	)?;

	let get_avg_color = |mask: &Mat| -> Result<core::Vec3b> {
		let mean_val = core::mean(&blurred, mask)?;
		let b = mean_val[0].clamp(0.0, 255.0).round() as u8;
		let g = mean_val[1].clamp(0.0, 255.0).round() as u8;
		let r = mean_val[2].clamp(0.0, 255.0).round() as u8;
		Ok(core::Vec3b::from([b, g, r]))
	};

	let start_bgr = if core::count_non_zero(&start_mask)? > 0 {
		get_avg_color(&start_mask)?
	} else {
		core::Vec3b::all(0)
	};

	let end_bgr = if core::count_non_zero(&end_mask)? > 0 {
		get_avg_color(&end_mask)?
	} else {
		core::Vec3b::all(0)
	};

	let start_hex = format!(
		"#{:02x}{:02x}{:02x}",
		start_bgr[2], start_bgr[1], start_bgr[0]
	);
	let end_hex = format!("#{:02x}{:02x}{:02x}", end_bgr[2], end_bgr[1], end_bgr[0]);

	Ok(GradientResult {
		start_color: start_hex,
		end_color: end_hex,
		angle: angle,
	})
}
