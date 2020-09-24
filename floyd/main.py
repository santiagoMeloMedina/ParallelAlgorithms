
import os
import input as Data
import floyd as Floyd
import check as Check

LIMIT = 500

def main():
    try: os.mkdir("./results")
    except: pass
    name = input("Study Name: ")
    try:
        os.mkdir("./results/{}".format(name))
        Gs, higher = Data.choose(LIMIT)
        results = []
        for n in range(len(Gs)):
            results.append(Floyd.GPU(Gs[n], higher))
            file = open("./results/{}/{}.txt".format(name, n), "w")
            file.write("Project {} #{} -- Size: {}\n\n".format(name, n, min(LIMIT, len(results[n][0]))))
            file.write(str(results[n][0]).replace("], ", "],\n"))
            file.close()
        Check.check(results)
    except:
        print("An error ocurred or project already exists")
        if (int(input("Would you like to delete the project with the name '{}'? |0|1|: ".format(name)))):
            os.rmdir("./results/{}".format(name))
    return

main()
#Floyd.comparison(int(input()))
