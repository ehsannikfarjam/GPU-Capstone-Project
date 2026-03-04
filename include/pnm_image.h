#ifndef PNM_IMAGE_H
#define PNM_IMAGE_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <stdexcept>

struct Pixel {
    unsigned char r, g, b;
};

class PNMImage {
public:
    int width;
    int height;
    int maxVal;
    bool isColor;
    std::vector<unsigned char> data; // For PGM (Grayscale)
    std::vector<Pixel> pixels;      // For PPM (Color)

    PNMImage() : width(0), height(0), maxVal(255), isColor(false) {}

    // Load PPM or PGM
    void load(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) throw std::runtime_error("Cannot open file: " + filename);

        std::string magic;
        file >> magic;

        if (magic == "P5") {
            isColor = false;
        } else if (magic == "P6") {
            isColor = true;
        } else {
            throw std::runtime_error("Unsupported PNM format (only P5/P6 supported): " + magic);
        }

        // Skip comments
        char c;
        file >> std::ws;
        while (file.peek() == '#') {
            std::string comment;
            std::getline(file, comment);
            file >> std::ws;
        }

        file >> width >> height >> maxVal;
        file.get(); // skip single whitespace

        if (isColor) {
            pixels.resize(width * height);
            file.read(reinterpret_cast<char*>(pixels.data()), width * height * 3);
        } else {
            data.resize(width * height);
            file.read(reinterpret_cast<char*>(data.data()), width * height);
        }
        file.close();
    }

    // Save as PPM or PGM
    void save(const std::string& filename) {
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) throw std::runtime_error("Cannot open file for writing: " + filename);

        if (isColor) {
            file << "P6\n" << width << " " << height << "\n" << maxVal << "\n";
            file.write(reinterpret_cast<const char*>(pixels.data()), width * height * 3);
        } else {
            file << "P5\n" << width << " " << height << "\n" << maxVal << "\n";
            file.write(reinterpret_cast<const char*>(data.data()), width * height);
        }
        file.close();
    }
};

#endif // PNM_IMAGE_H
