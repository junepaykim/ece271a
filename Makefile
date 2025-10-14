LATEX = pdflatex
LATEXFLAGS = -interaction=nonstopmode -halt-on-error -file-line-error

HWS_EXIST := $(patsubst %/,%,$(wildcard hw*/))
HWS_ALL := $(HWS_EXIST)

# hwX.pdf를 만드는 “명시적 규칙”을 자동 생성 (패턴 규칙 의존 X)
define MAKE_ONE
$1.pdf: main.tex $1/$1.tex
	@echo "Building $1.pdf..."
	$(LATEX) $(LATEXFLAGS) -jobname=$1 "\def\singlehw{$1}\input{main.tex}"
	$(LATEX) $(LATEXFLAGS) -jobname=$1 "\def\singlehw{$1}\input{main.tex}"
endef
$(foreach hw,$(HWS_EXIST),$(eval $(call MAKE_ONE,$(hw))))

# 별칭: make hw1 -> hw1.pdf
.PHONY: $(HWS_ALL)
$(HWS_ALL): %: %.pdf

clean:
	rm -f *.aux *.log *.out *.toc *.pdf
	rm -f $(foreach hw,$(HWS_ALL),$(hw).pdf)
