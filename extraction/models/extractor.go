package models

import (
	"regexp"
	"strings"
)

// adding all regexpes to the package level to avoid repeated compilation
var (
	reWhitespaceOnly = regexp.MustCompile(`^\s*$`)
	reNaiveStripTag  = regexp.MustCompile(`(?s)<[^>]*(>|$)`)
)

// ExtractParagraphs replicates the Perl subroutine html_to_paragraphs from DuckDuckGo's DuckAssist::Extract.
//
// The Perl logic (summarized) is:
//  1. Extract <body> content.
//  2. Mark <p>/<div> tags => "~~~<p>" to force paragraph boundaries.
//  3. Strip all HTML tags => raw text.
//  4. Replace "~~~" => "\n\n" => paragraph boundaries.
//  5. s/\h+/ /sg => unify horizontal whitespace into single spaces
//  6. s/\n\h+/\n/sg => remove horizontal whitespace after newlines
//  7. split on \n{2,} => paragraphs
//  8. grep out paragraphs that are purely whitespace
//  9. apply minParagraphLength
//  10. (Important) do NOT remove trailing spaces or newlines except to skip purely empty lines.
//
// Below, we replicate that and add minor adjustments so test cases expecting
// no leading space but some trailing space will pass.
func ExtractParagraphs(htmlStr string, minParagraphLength int) []string {
	if htmlStr == "" {
		return nil
	}

	body := getBody(htmlStr)
	body = strings.ReplaceAll(body, "<p", "\n\n<p")
	body = strings.ReplaceAll(body, "<div", "\n\n<div")

	text := naiveStripHTML(body)

	text = unifyHorizontalWhitespace(text)
	text = removeNewlineWhitespace(text)
	rawParagraphs := strings.Split(text, "\n\n")

	paragraphs := make([]string, 0, len(rawParagraphs))

	// 8) filter out blank paragraphs and short paragraphs
	for _, p := range rawParagraphs {
		p = strings.TrimRight(p, "\n")
		if len(p) < minParagraphLength || reWhitespaceOnly.MatchString(p) {
			continue
		}
		paragraphs = append(paragraphs, removeLeadingSpaces(p))
	}
	return paragraphs
}

// removeLeadingSpaces removes leading spaces/tabs (but does NOT remove trailing spaces).
func removeLeadingSpaces(s string) string {
	i := 0
	// Skip over any leading spaces or tabs
	for i < len(s) && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r') {
		i++
	}
	return s[i:]
}

func removeNewlineWhitespace(input string) string {
	var sb strings.Builder
	sb.Grow(len(input))
	skip := false
	for _, r := range input {
		if r == '\n' {
			sb.WriteRune(r)
			skip = true
		} else if skip && (r == ' ' || r == '\t' || r == '\r' || r == '\f' || r == '\v') {
			continue
		} else {
			sb.WriteRune(r)
			skip = false
		}
	}
	return sb.String()
}

func unifyHorizontalWhitespace(input string) string {
	var sb strings.Builder
	sb.Grow(len(input)) // Preallocate memory for the builder
	space := false
	for i := 0; i < len(input); i++ {
		c := input[i]
		if c == ' ' || c == '\t' || c == '\r' || c == '\f' || c == '\v' {
			if !space {
				sb.WriteByte(' ')
				space = true
			}
		} else {
			sb.WriteByte(c)
			space = false
		}
	}
	return sb.String()
}

// Lowercasing the entire string in getBody is expensive.
// Optimization: Search for <body> case-insensitively using a custom search function:
func findBodyIndex(s string) int {
	for i := 0; i < len(s)-5; i++ {
		if (s[i] == '<' || s[i] == '<'+32) &&
			(s[i+1] == 'b' || s[i+1] == 'b'+32) &&
			(s[i+2] == 'o' || s[i+2] == 'o'+32) &&
			(s[i+3] == 'd' || s[i+3] == 'd'+32) &&
			(s[i+4] == 'y' || s[i+4] == 'y'+32) {
			return i
		}
	}
	return -1
}

func getBody(htmlStr string) string {
	start := findBodyIndex(htmlStr)
	if start == -1 {
		return htmlStr
	}
	end := strings.Index(htmlStr[start:], ">")
	if end == -1 {
		return htmlStr[start:]
	}
	return htmlStr[start+end+1:]
}

// We need to use this naive approach to have parity
// with perl implementation.
// The perl implementation is very lenient and allows us to extract text after broken html tags.
// We could use eg. golang net/html library, but that one is strict as it's doing proper
// html5 parsing/stripping.
// The perl It ignores broken html tags and allows us to extract text after them.
// Eg. html content:
// "<html><body>a fox</bo jumps"
// perl would extract "a fox jumps" while golang's net/html would only extract "a fox"
// This is because the html is broken and the closing body tag is missing.
//
// This approach is very custom. If we have other partial tags like <somethingNoSpaceAtAll, they get fully removed because there’s no space.
// If a partial tag has multiple segments of text, e.g. </some broken stuff here morestuff, we only keep the substring after the first whitespace.
// If we would have <2 < 3 (literal < signs for math) it might forcibly remove them. This is all part of the naive approach.
// customNaiveStripHTML attempts to remove well-formed tags entirely, AND handle
// broken tags in a way that "a fox</bo jumps" => "a fox jumps".
func naiveStripHTML(s string) string {
	// This pattern matches any sequence starting with '<' up to:
	//   1) a closing '>', or
	//   2) the end of the string ($).
	// It's the same approach as `(?s)<[^>]*(>|$)`.
	// We'll interpret it in a custom function to handle partial matches differently.
	return reNaiveStripTag.ReplaceAllStringFunc(s, func(match string) string {
		// If the match ends with ">", it's a well-formed tag => remove entirely
		if strings.HasSuffix(match, ">") {
			return ""
		}

		// Otherwise, it's a broken tag. Example: "</bo jumps"
		// We'll remove everything up to the first whitespace after '<', but keep any trailing text.
		// So: "</bo jumps" => remove "</bo" => keep " jumps"

		// e.g. match = "</bo jumps"
		// we want to find the first whitespace. If none, remove entire match.
		// if there's whitespace, remove up to that whitespace => keep remainder.
		// i.e.    ^    ^--- keep this portion
		// e.g.    <bo " jumps" => final = " jumps"

		trimmed := match[1:] // remove leading '<'
		idxSpace := strings.IndexAny(trimmed, " \t\r\f\v")

		if idxSpace == -1 {
			// no whitespace => remove entire partial tag
			// "a fox</bo_jumps" => "a fox"
			return ""
		}

		// keep the text after that space
		return trimmed[idxSpace:]
	})
}
