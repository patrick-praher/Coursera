sigm <- function(x){
  return(1/(1+exp(-x)))
}
curve(sigm, from = -10, to=10)


x <- c(2.5, 0.3, 2.8, 0.5)
y <- c(1, 0, 1, 1)

prob <- sapply(x,sigm)

prob[!y]<- 1-prob[!y]

prod(prob)
#0.23

prob <- sapply(x,sigm)
sum((y - prob)*x)
#0.37