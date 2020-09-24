
def check(Gs):
    response = int(input("Which subgraph, there are 0-{}: ".format(len(Gs)-1)))
    while response >= 0 and response <= len(Gs)-1:
        source = int(input("Source from 0 to {}: ".format(len(Gs[response][0])-1)))
        target = int(input("Target from 0 to {}: ".format(len(Gs[response][0])-1)))
        print("Minimum distance from {} to {}: ".format(source, target), Gs[response][0][source][target])
    return
