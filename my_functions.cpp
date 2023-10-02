#include "Halide.h"
#include <stdio.h>
using namespace Halide;

int main() {
    {
        Func brighter;
        Var x, y, c;

        // Simple float value
        Param<float> factor;

        // AOT version of a buffer, instead of passing in just an input into this function, it will require an
        // output buffer as an argument with identical dimensions of the input buffer (use CloneDims(input))
        ImageParam input(type_of<float>(), 3); // The 3 means 3 dimensional
        Expr value = input(x, y, c) * factor; // Multiply by brightness factor
        brighter(x, y, c) = min(value, 1); // Round down if R/G/B value went over 1

        // The function is called "halide_brighter"
        brighter.compile_to_static_library("comp/adjust_brightness", {input, factor}, "halide_brighter");
    }

    {
        Func invert;
        Var x, y, c;
        ImageParam input(type_of<float>(), 3);
        Expr value = 1 - input(x, y, c);
        invert(x, y, c) = value;
        invert.compile_to_static_library("comp/invert", {input}, "halide_invert");
    }

    {
        Func rgb_to_gray;
        Var x, y;
        ImageParam input(type_of<float>(), 3);
        rgb_to_gray(x, y) = (0.2989f * input(x, y, 0)) + (0.587f * input(x, y, 1)) + (0.114f * input(x, y, 2));
        rgb_to_gray.compile_to_static_library("comp/rgb_to_grayscale", {input}, "halide_grayscale");

    }

    {
        Func hflip;
        Var x, y, c;
        ImageParam input(type_of<float>(), 3);
        hflip(x, y, c) = input(input.dim(0).max()-x, y, c);
        hflip.compile_to_static_library("comp/hflip", {input}, "halide_hflip");
    }

    {
        Func vflip;
        Var x, y, c;
        ImageParam input(type_of<float>(), 3);
        vflip(x, y, c) = input(x, input.dim(1).max()-y, c);
        vflip.compile_to_static_library("comp/vflip", {input}, "halide_vflip");
    }

    {
        Func replace;
        Var x, y, c;
        Param<int> i;
        Param<int> j;
        Param<int> h;
        Param<int> w;
        ImageParam input(type_of<float>(), 3);
        ImageParam other(type_of<float>(), 3);

        replace(x, y, c) = other(x, y, c);
        Func erase;
        //  constant_exterior(source, value, x_start, x_end, y_start, y_end
        //  repeat_edge better?
        //  select better?
        erase(x, y, c) = BoundaryConditions::constant_exterior(replace, input(x, y, c), i, i+w, j, j+h)(x, y, c);

        erase.compile_to_static_library("comp/erase", {input, i, j, h, w, other}, "halide_erase");
    }

    {
        Func replace;
        Var x, y, c;
        Param<int> i;
        Param<int> j;
        Param<int> h;
        Param<int> w;
        Param<float> v;
        ImageParam input(type_of<float>(), 3);

        replace(x, y, c) = input(x, y, c) * 0;
        Func erase;
        erase(x, y, c) = BoundaryConditions::constant_exterior(replace, input(x, y, c), i, i + w, j, j + h)(x, y, c);

        erase.compile_to_static_library("comp/erase_2", {input, i, j, h, w, v}, "halide_erase_2");
    }

    {
        Func solarize;
        Func invert;
        Param<float> threshold;
        Var x, y, c;
        ImageParam input(type_of<float>(), 3);
        invert(x, y, c) = 1 - input(x, y, c);
        solarize(x, y, c) = select(input(x, y, c) >= threshold, invert(x, y, c), input(x, y, c));
        solarize.compile_to_static_library("comp/solarize", {input, threshold}, "halide_solarize");
    }

    {
        try {
            Func autocontrast("autocontrast");
            ImageParam input(type_of<float>(), 3);
            Param<float> min_r;
            Param<float> min_g;
            Param<float> min_b;
            Param<float> scale_r;
            Param<float> scale_g;
            Param<float> scale_b;
            Var x, y, c;
            Expr value;

            value = clamp(select(c == 0, ((input(x, y, 0) - min_r) * scale_r),
                           c == 1, ((input(x, y, 1) - min_g) * scale_g),
                           ((input(x, y, 2) - min_b) * scale_b)), 0, 1);
            autocontrast(x, y, c) = value;
            autocontrast.compile_to_static_library("comp/autocontrast", {input, min_r, min_g, min_b, scale_r, scale_g, scale_b},
                                                   "halide_autocontrast");
        } catch (CompileError &err) {
            std::cerr << "EXCEPTION!!! " << err.what() << "\n";
        }
    }

    {
        Func crop;
        Var x, y, c;
        Param<int> i;
        Param<int> j;
        Param<int> h;
        Param<int> w;
        ImageParam input(type_of<float>(), 3);

        Func replace;
        replace(x, y, c) = input(x + i, y + j, c);

        replace.compile_to_static_library("comp/crop", {input, i, j, h, w}, "halide_crop");
    }

    {
        Func adjust_gamma;
        Var x, y, c;
        Param<float> gamma;
        Param<float> gain;
        ImageParam input(type_of<float>(), 3);
        Expr value = gain * (pow(input(x, y, c), gamma)); //  The std::pow function is extremely slow, especially with
                                                          //  a float value, must figure out how to speed this up as it
                                                          //  is way slower than the python implementation
        adjust_gamma(x, y, c) = clamp(value, 0, 1);

        adjust_gamma.compile_to_static_library("comp/adjust_gamma", {input, gamma, gain}, "halide_adjust_gamma");
    }

    {
        Func adjust_saturation;
        Var x, y, c;
        Param<float> saturation;
        ImageParam input(type_of<float>(), 3);

        Func rgb_to_gray;
        rgb_to_gray(x, y) = (0.2989f * input(x, y, 0)) + (0.587f * input(x, y, 1)) + (0.114f * input(x, y, 2));
        adjust_saturation(x, y, c) = clamp((saturation * input(x, y, c) + (1 - saturation) * rgb_to_gray(x, y)),
                                           0, 1);

        adjust_saturation.compile_to_static_library("comp/adjust_saturation", {input, saturation},
                                                    "halide_adjust_saturation");
    }

    {
        Func adjust_contrast("adjust_contrast");
        Var x, y, c;
        Param<float> contrast;
        Param<float> mean;
        ImageParam input(type_of<float>(), 3);

        adjust_contrast(x, y, c) = clamp((contrast * input(x, y, c) + (1 - contrast) * mean), 0, 1);

        adjust_contrast.compile_to_static_library("comp/adjust_contrast", {input, contrast, mean},
                                                  "halide_adjust_contrast");
    }

    {
        Func normalize("normalize");
        Var x, y, c;
        Param<float> contrast;
        Param<float> r_mean;
        Param<float> b_mean;
        Param<float> g_mean;
        Param<float> r_sd;
        Param<float> g_sd;
        Param<float> b_sd;
        ImageParam input(type_of<float>(), 3);

        Expr value;
        Func r;
        Func g;
        Func b;
        r(x, y, c) = (input(x, y, 0) - r_mean)/r_sd;
        g(x, y, c) = (input(x, y, 1) - g_mean)/g_sd;
        b(x, y, c) = (input(x, y, 2) - b_mean)/b_sd;
        value = clamp(select(c == 0, r(x, y, c),
                       c == 1, g(x, y, c),
                       b(x, y, c)), 0, 1);

        normalize(x, y, c) = value;

        normalize.compile_to_static_library("comp/normalize", {input, r_mean, g_mean, b_mean, r_sd, g_sd, b_sd},
                                                  "halide_normalize");
    }

    {
        Func posterize("posterize");
        Var x, y, c;
        Param<uint8_t> mask;
        ImageParam input(type_of<uint8_t>(), 3);
        posterize(x, y, c) = input(x, y, c) & mask;

        posterize.compile_to_static_library("comp/posterize", {input, mask}, "halide_posterize");
    }

    {
        try {
            Func elastic_transform("elastic_transform");
            Var x,y,c,d;
            ImageParam input(type_of<float>(), 3);
            ImageParam displacement(type_of<float>(), 4);
            Param<float> fill("fill", 0.0f);

            //Calculate the pixel bias from the original image
            Expr near_x = cast<int>(round( displacement(0,x,y,0) * input.width() /2));
            Expr near_y = cast<int>(round( displacement(1,x,y,0) * input.height() /2));

            //extend image with border filled with zero
            Func extended_image = BoundaryConditions::constant_exterior(input,fill);

            elastic_transform(x,y,c) = extended_image(x + near_x, y + near_y, c);

            elastic_transform.compile_to_static_library("comp/elastic_transform", {input, displacement, fill}, "halide_elastic_transform");

            } catch(CompileError &err) {
            std::cerr << "EXCEPTION!!! " << err.what() << "\n";
        }

    }


    {
        Func resize("resize");
        Var x, y, c;
        ImageParam input(type_of<float>(), 3);
        Param<float> scale_x("scale_x"), scale_y("scale_y");
        
        Expr src_x = x / scale_x;
        Expr src_y = y / scale_y;

        Expr x1 = cast<int>(floor(src_x));
        Expr y1 = cast<int>(floor(src_y));
        Expr x2 = x1 + 1;
        Expr y2 = y1 + 1;

        Expr a = src_x - x1;
        Expr b = src_y - y1;

        Expr p1 = (1 - a) * (1 - b) * input(x1, y1, c);
        Expr p2 = a * (1 - b) * input(x2, y1, c);
        Expr p3 = (1 - a) * b * input(x1, y2, c);
        Expr p4 = a * b * input(x2, y2, c);

        resize(x, y, c) = cast<float>(p1 + p2 + p3 + p4);
        resize.compile_to_static_library("comp/resize", {input, scale_x, scale_y},
                                                      "halide_resize");
    }

    {
        // to adjust hue we need to convert rgb to hsv, adjust hue and convert back
        Func adjust_hue;
        Var x, y, c; 
        Param<float> hue_factor;
        ImageParam input(type_of<float>(), 3);
 
        Expr r = input(x, y, 0);
        Expr g = input(x, y, 1);
        Expr b = input(x, y, 2);

        Expr c_max = max(r, g, b);
        Expr c_min = min(r, g, b);
        Expr delta = c_max - c_min;

        // calculate hue 
        Expr hue = select(
            delta == 0, 0,
            c_max == r, ((g - b) / delta) % 6,
            c_max == g, (b - r) / delta + 2,
            (r - g) / delta + 4
        ) * 60;

        // calculate saturation
        Expr saturation = select(
            c_max == 0, 0,
            delta / c_max
        );

        // calculate value
        Expr value = c_max;

        // rgb to hsv func
        Func hsv;
        hsv(x, y, c) = select(
            c == 0, hue,
            c == 1, saturation,
            value
        );

        // adjust hue values
        Func hsv_adjust_hue;
        hsv_adjust_hue(x, y, c) = select(
            c == 0, hsv(x, y, 0) + hue_factor * 360,
            hsv(x, y ,c)
        );

        Func clamped;
        Halide::Expr clamped_value = hsv_adjust_hue(x, y, c);
        clamped_value = select(
            clamped_value > 360, clamped_value - 360,
            clamped_value < 0, clamped_value + 360,
            clamped_value
        );

        clamped(x, y, c) = clamped_value;

        Expr H = clamped(x, y, 0);
        Expr S = clamped(x, y, 1);
        Expr V = clamped(x, y, 2);

        Expr C = S * V;
        Expr X = C * (1 - abs(operator%(H / 60, 2) - 1));
        Expr m = V - C;

        Expr R1, G1, B1;

        R1 = select(
            H < 60 || H >= 300, C,
            H < 120 || H >= 240, X,
            0
        );

        G1 = select(
            H >= 60 && H < 180, C,
            H >= 240, 0,
            X
        );

        B1 = select(
            H < 120, 0,
            H >= 180 && H < 300, C,
            X
        );

        Func rbg_to_hsv;
        rbg_to_hsv(x, y, c) = select(
            c == 0, R1 + m,
            c == 1, G1 + m,
            B1 + m
        );

        adjust_hue(x, y, c) = rbg_to_hsv(x, y, c);
        adjust_hue.compile_to_static_library("comp/adjust_hue", {input, hue_factor}, "halide_adjust_hue");
    }
    // {
    //     Func perspective;
    //     ImageParam input(type_of<float>(), 3);
    //     // s1, s2, s3, s3 -> top-left, top-right, bottom-right, bottom-left
    //     Param<int> s1_x, s1_y, s2_x, s2_y, s3_x, s3_y, s4_x, s4_y;
    //     Param<int> e1_x, e1_y, e2_x, e2_y, e3_x, e3_y, e4_x, e4_y;

    //     // GeneratorInput<>

    //     // perspective.compile_to_static_library("comp/perspective", {input, s1_x, s1_y, s2_x, s2_y, s3_x, s3_y, s4_x, s4_y, e1_x, e1_y, e2_x, e2_y, e3_x, e3_y, e4_x, e4_y},
    //     //                                       "halide_perspective");
    // }

    {
        Func adjust_sharpness;
        ImageParam input(Float(32), 3);

        Param<float> sharpness_factor; 
        Var x, y, c;
        
        float kernel[3][3] = {
            {1.0f/13.0f, 1.0f/13.0f, 1.0f/13.0f},
            {1.0f/13.0f, 5.0f/13.0f, 1.0f/13.0f},
            {1.0f/13.0f, 1.0f/13.0f, 1.0f/13.0f}
        };

        // float kernel[3][3] = {
        //     {0.0769, 0.0769, 0.0769},
        //     {0.0769, 0.3846, 0.0769},
        //     {0.0769, 0.0769, 0.0769}
        // };

        Func edge_repeated;
        edge_repeated = BoundaryConditions::repeat_edge(input);


        Expr sum = 0.0f;
        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                sum = operator+(sum, (edge_repeated(x + i, y + j, c) * kernel[i + 1][j + 1]));
            }
        }

        Func filter_applied;
        filter_applied(x, y, c) = sum;
        
        Func blended;
        blended(x, y, c) = clamp(sharpness_factor * edge_repeated(x, y, c) + (1.0f - sharpness_factor) * filter_applied(x, y, c), 0.0f, 1.0f);

        adjust_sharpness(x, y, c) = select(
            x == 0 || x == (input.width() - 1), input(x, y, c),
            y == 0 || y == (input.height() - 1), input(x, y, c),
            blended(x, y, c)
        );

        adjust_sharpness.compile_to_static_library("comp/adjust_sharpness", {input, sharpness_factor}, "halide_adjust_sharpness");
    }
//     Handy error catching block - thanks Chris and Adam
//     try {
//            } catch(CompileError &err) {
//            std::cerr << "EXCEPTION!!! " << err.what() << "\n";
//        }
}
