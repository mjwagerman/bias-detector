from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

text = "Ukrainian President Volodymyr Zelensky said his Oval Office meeting last week with US President Donald Trump did not go as expected, describing it as regrettable and emphasizing that Ukraine is ready to negotiate an end to the conflict. He reiterated Ukraine’s commitment to peace, addressing the meeting directly on X. During the meeting, Trump and Vice President JD Vance accused Zelensky of gambling with World War Three and warned him that his country was in big trouble. Zelensky stated that the meeting at the White House was not as intended and expressed hope for future constructive cooperation. He also participated in the Securing Our Future Summit on Ukraine and European security in London on March 2, 2025. Western leaders hope his statement will help mend relations with the White House, but the tense Oval Office meeting highlighted the strain between Kyiv and Washington. Zelensky also announced that Ukraine is ready to sign a minerals deal that was supposed to be finalized on Friday before the argument led to his early departure. He emphasized that Ukraine sees the agreement as a step toward greater security and strong guarantees. Additionally, he outlined a possible ceasefire framework, including the release of prisoners, a truce in the sky banning missiles and long-range drones, and an immediate ceasefire at sea if Russia reciprocates. He expressed a desire to move quickly through the next stages and work with the US on a final agreement. The proposed framework aligns with a plan suggested by French President Emmanuel Macron following a summit of Western leaders in London. Zelensky acknowledged America’s past support in helping Ukraine maintain its sovereignty, recalling Trump’s provision of Javelins as a pivotal moment. It remains uncertain how Trump will respond to Zelensky’s proposals or his reflections on the White House visit, but the statement suggests Kyiv is pushing to have a stronger voice in discussions about the conflict’s future, especially after the Trump administration began talks with Russia last month without inviting Ukraine."

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

model = AutoModelForSequenceClassification.from_pretrained("bucketresearch/politicalBiasBERT")


inputs = tokenizer(text, return_tensors="pt")
labels = torch.tensor([0])
outputs = model(**inputs, labels=labels)
loss, logits = outputs[:2]

# [0] -> left 
# [1] -> center
# [2] -> right
print(logits.softmax(dim=-1)[0].tolist())
