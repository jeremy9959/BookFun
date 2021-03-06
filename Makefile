CHAPTERS = $(shell find chapters/ -type f -name '*.md' | sort)
FILTERS = --filter pandoc-xnos --citeproc
OPTIONS = -N --standalone --mathjax --toc --top-level-division=chapter
METADATA = --metadata-file metadata.yml
OUTPUT = -o build.pdf


book: build.pdf

build.pdf : $(CHAPTERS)
	cat $(CHAPTERS) | pandoc $(OPTIONS) $(FILTERS) $(METADATA) $(OUTPUT)


