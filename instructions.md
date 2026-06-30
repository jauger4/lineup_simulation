## Overview
 recently I was having a conversation with a friend of mine about where to put guys in a lineup to maximize run potential. Some interesting points were brought up and we questioned the traditional lineup-making procedure where you put a fast contact guy first, someone that's well-rounded second, and then some more powerful guys third, fourth, and then the rest of your guys mixed in after.

When you think about it there might be other ways to do this. Putting a power hitter in the middle of the order could be counterproductive considering power hitters typically are looking for true outcomes, where the ball is not put into play as much as a contact hitter would be putting the ball into play. Power hitters are looking to hit the ball far and be selective, thus resulting in more strikeouts and more walks.

Would it make more sense to put a power hitter earlier in the lineup, as a walk or extra bases may be more likely, with contact hitters coming up behind to potentially move them over or in to home plate? To test this I want to look at some StatCast data to cluster hitters based off of their approach and run Monte Carlo simulations to test the hypothesis of having power hitters higher in the order as opposed to in the middle.

## Clustering
 StatCast data can tell you a lot about what a hitter is trying to do at the plate. Using this data we can cluster hitters based off of their approaches, giving us more insight as to where to put them in the lineup to successfully test this hypothesis. I would also like to be able to see which teams are using traditional lineups and which ones may have already been testing this hypothesis in real time. This will give me good insight into which lineups to actually simulate and which ones I can look past for this testing. 

The numbers that we will look at for the cluster rate are going to be:

- Strikeout percentage
- Walk percentage
- Chase percentage
- Whiff percentage
- Barrel percentage

These five numbers should give us a really good insight into what hitters are trying to accomplish at the plate. Your typical power hitter will have a high strikeout rate and a high whiff rate, low chase and high walk as well as high barrel. Contact hitters, on the other hand, will chase a lot but not whiff much as well as not strike out a lot or walk a lot. They are trying to put the ball into play. Barrels are not as important. They are just trying to slap something into the field of play as opposed to do damage on every swing. 

The algorithms that I would like to use to cluster here are:

- K-means
- DBSCAN
- Gaussian Mixture Model (GMM)

These are all slightly different clustering machine learning models. I would like to test this process with all three and see which gives the best results.

## Monte Carlo Simulation
 Once we are able to determine which hitters are looking to do damage at the plate and which hitters are looking to put the ball into play, we can now look at lineups and see which ones would benefit our hypothesis testing. We will run a Monte Carlo simulation to determine if moving power hitters up in the lineup can provide statistically significant results showing that the team would score more runs with that lineup as opposed to their current lineup.

This test should be able to simulate, based off of batted ball events, such as:

- strikeouts
- walks
- hits
- extra base hits
- home runs
- outs
- double plays
- sacrifices
- et cetera

This will be run by reorganizing lineups to have power hitters at the top and contact hitters in the middle of the lineup.

Lastly once this simulation has been done, we must check to see if the means from runs per game in the current lineups that they are using in real life and the means from the simulation show a statistically significant difference.