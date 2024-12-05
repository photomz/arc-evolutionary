I think ARC-AGI is the most important benchmark we have today. It’s surprising that even the most sophisticated Large Language Models (LLMs), like OpenAI o1 and Claude Sonnet 3.5, struggle with simple puzzles that humans can solve easily.

This highlights the core limitation of current LLMs: they're bad at reasoning about things they weren't trained on. They are bad at generalizing.

After reading Ryan Greenblatt’s blog post on how he achieved a state of the art 43% accuracy on ARC-AGI-Pub, I wondered if we could push these models further. Could it be that frontier models might actually possess the necessary intelligence and understanding to solve ARC? Maybe if they are poked and prodded enough, they’ll spit out the right answer.

After lots of experimenting, I got a record of xx% on the public leaderboard using Sonnet 3.5. This is a significant improvement over the previous high score (Ryan’s) of 43%.

My approach works by having Sonnet 3.5 generate a bunch of Python transform functions, testing them against challenge examples, and then using the best-performing functions to create new prompts for generating even better solutions. This process repeats multiple times, ultimately generating up to 500 functions using 31 dynamic prompts per challenge.

I have the LLM generate Python functions, instead of just outputting solution grids, because functions can be executed and verified for correctness (detailed in the next section) but grids cannot.

While this approach is computationally intensive compared to human intuitive reasoning, it may hint at a path forward. LLMs can compensate for their generalization limitations through scaled test-time compute guided by evolutionary principles. By generating waves of candidate solutions and using their performance to guide further compute allocation, we can efficiently explore the solution space even with limited prior knowledge. It's possible that this kind of test-time compute scaling is how Artificial General Intelligence (AGI) emerges.

The key to solving ARC may be the key to solving AGI.
