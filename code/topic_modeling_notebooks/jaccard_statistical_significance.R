library(jaccard)

args <- commandArgs(trailingOnly = TRUE)
# comment jaccard.test.mca(as.list(strsplit("1, 0, 1", ", ")[[1]]), as.list(strsplit("1, 0, 0", ", ")[[1]]))
vec1 <- as.numeric(unlist(strsplit(substr(args[1],3,nchar(args[1])-1), ",")))
vec2 <- as.numeric(unlist(strsplit(substr(args[2],3,nchar(args[2])-1), ",")))
print(jaccard.test.bootstrap(vec1, vec2))