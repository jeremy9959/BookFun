# Logistic Regression

Suppose that we are trying to convince customers to buy our product by showing them advertising.  Our experience teaches
us that there is no deterministic relationship between how often a potential customer sees one of our ads and whether or not
they purchase our product, nevertheless it is the case that as they see more ads they become more likely to make a purchase.
Logistic regression is a statistical model that can capture the essence of this idea.

To make this problem more abstract, let's imagine that we are trying to model a random event that depends on a parameter.
As in our introduction above, the random event might be a user deciding to make  a purchase from a website, which, in our very simple model,
depends on how many times the user saw an advertisement for the product in question.  But we could imagine other situations where
the chance of an event happening depends on a paramter.  For example, we could imagine that a student's score on a certain test depends on how much studying they
do, with the likelihood of passing the test increasing with the amount of studying.  

To construct this model, we assume that the probability of a certain event $p$ is related to some parameter $x$ by the following relationship:

$$
\log\frac{p}{1-p} = ax+b 
$${#eq:logistic_1}

where $a$ and $b$ are constants.  The quantity $\frac{p}{1-p}$ is the "odds" of the event occurring.  We often use this quantity colloquially; if
the chance of our team winning a football game is $1$ in $3$, then we would say the odds of a win are $1$-to-$2$, which we can interpret as meaning they 
are twice as likely to lose as to win.  The quantity $\log\frac{p}{1-p}$ is, for obvious reasons, called the log-odds of the event.

The assumption in @eq:logistic_1 can be written
$$
\frac{p}{1-p} = e^{ax+b}
$$
and we interpret this as telling us that if the parameter $x$ increases by $1$, the odds of our event happening go up by a factor of $e^{a}$. 
So, to be even more concrete, if $a=\log 2$, then our logistic model would say that an increase of $1$ in our parameter $x$ doubles the odds of
our event taking place. 

In terms of the probability $p$, equation @eq:logistic_1 can be rewritten
$$
p = \frac{1}{1+e^{-ax-b}}
$$
This proposed relationship between the probability $p$ and the parameter $x$ is called the *logistic model.*  The function
$$
\sigma(x) = \frac{1}{1+e^{-x}}
$$
is called the *logistic function* and yields an S-shaped curve. 

![Logistic Curve](img/logistic_curve.png){#fig:logistic_curve width=50%}

To fully put the logistic model in perspective, let's choose some explicit parameters and look at what data arising from such a model
would look like.  Imagine therefore that $a=\log 2$ and $b=0$, so that the probability of the event we are interested occurring is given by the
formula
$$
p(x) = \frac{1}{1+e^{-(\log 2)x}} = \frac{1}{1+(.5)^x}.
$$
Our data consists of counts of how often our event happened for a range of values of $x$.  To generate this data, we can pick $x$ values from the
set $\{-3,-2,-1,0,1,2,3\}$ yielding probabilities $\{.11,.2,.33,.4,.56,.67,.8\}$.  Now our data consists of,  for each value of $x$, the result of $100$ independent
Bernoulli trials with probability $p(x)$.  For example, we might find that our event occurred $\{10, 18, 38, 50, 69, 78, 86\}$ times respectively for each of the $x$ values.


## Fitting the logistic model via Maximum Likelihood

In applications, our goal is to choose the parameters of a logistic model to accurately predict the likelihood of the event under study occurring as a function
of the measured parameter.  Let's imagine that we collected the data that we generated above, without knowing that it's source was a logistic model.  So
@tbl:logistic_data shows the number of times the event occurred, for each of the measured values of the $x$ parameter.

|$x$ |-3 | -2 | -1 | 0 | 1 | 2 | 3 |
|---|---|---|---|---|---|---|---|
|Occurrences (out of 100)|10  |18 | 38 | 50 | 69 | 78 | 86|

Table: Sample Data {#tbl:logistic_data}

Our objective now is to find a logistic model which best explains this data. Concretely, we need to estimate the coefficients $a$ and $b$ that yield 
$$
p(x) = \frac{1}{1+e^{-ax-b}}
$$
where the resulting probabilities best estimate the data. As we have seen, this notion of "best" can have different interpretations.
For example, we could approach this from a Bayesian point of view, adopt a prior distribution on the parameters $a$ and $b$, and use the data to obtain
this prior and obtain a posterior distribution on $a$ and $b$.  For this first look at logistic regression, we will instead adopt a "maximum likelihood"
notion of "best" and ask what is the most likely choice of $a$ and $b$ to yield this data.

To apply the maximum likelihood approach, we need to ask "for (fixed, but unknown) values of $a$ and $b$, what is the likelihood that a logistic model
with those parameters would yield the data we have collected?  Each column in @tbl:logistic_data represents $100$ Bernoulli trials with a fixed probability
$p(x)$.  So, for example,  the chance $q$ of obtaining $10$ positive results with $x=-3$ is given by
$$
q(-3)=C p(-3)^{10}(1-p(-3))^{90}
$$
where $C$ is a constant (it would be a binomial coefficient).  Combining this for different values of $x$, we see that the likelihood of the data is
the product
$$
L = C' p(-3)^{10}(1-p(-3))^{90}p(-2)^{18}(1-p(-2))^{82}\cdots p(3)^{86}(1-p(3))^{14}
$$
where $C'$ is another constant.  Each $p(x)$ is a function of the parameters $a$ and $b$, so all together this is a function of those two parameters.
Our goal is to maximize it. 

One step that simplifies matters is to consider the logarithm of the likelihood:
$$
\log L = \sum_{i=0}^{6} \left[ x_{i}\log(p(x_{i})) + (100-x_{i})\log(1-p(x_{i}))\right]
$$






