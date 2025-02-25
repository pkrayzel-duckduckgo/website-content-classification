package main

import (
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"

	"extraction/models"
)

const minParagraphLength = 20 // Adjust as needed

func main() {
	if len(os.Args) != 3 {
		fmt.Println("Usage: go run main.go <input_directory> <output_directory>")
		os.Exit(1)
	}

	inputDir := os.Args[1]
	outputDir := os.Args[2]

	// Create output directory if it doesn't exist
	if err := os.MkdirAll(outputDir, os.ModePerm); err != nil {
		fmt.Printf("Failed to create output directory: %v\n", err)
		os.Exit(1)
	}

	// Read files from the input directory
	files, err := ioutil.ReadDir(inputDir)
	if err != nil {
		fmt.Printf("Failed to read input directory: %v\n", err)
		os.Exit(1)
	}

	for _, file := range files {
		if file.IsDir() {
			continue
		}

		inputFilePath := filepath.Join(inputDir, file.Name())
		outputFilePath := filepath.Join(outputDir, file.Name()+"_paragraphs.txt")

		// Read file content
		content, err := ioutil.ReadFile(inputFilePath)
		if err != nil {
			fmt.Printf("Failed to read file %s: %v\n", inputFilePath, err)
			continue
		}

		// Extract paragraphs
		paragraphs := models.ExtractParagraphs(string(content), minParagraphLength)

		// Write paragraphs to the output file
		outputContent := strings.Join(paragraphs, "\n\n")
		if err := ioutil.WriteFile(outputFilePath, []byte(outputContent), 0644); err != nil {
			fmt.Printf("Failed to write to file %s: %v\n", outputFilePath, err)
			continue
		}

		fmt.Printf("Processed file: %s\n", inputFilePath)
	}

	fmt.Println("Processing completed.")
}
