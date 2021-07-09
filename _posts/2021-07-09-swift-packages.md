---
layout: post
title:  "How Swift Packages work"
date:   2021-07-09 13:06:08 +0530
categories: swift
---

## Topics Covered
- Overview of **Swift Package Manager** and how to work with them (including **Unit Testing**).
- Explanation of **Maximum Pairwise Product** algorithm (coded in swift programming language).

## Intoduction
I've had my eye on the [*swift programming language*](https://swift.org/) for quite some time now and that was for two reasons:

1) Feels like *Python*.
2) Runs like *C/C++*.

These two qualities make *Swift* a compelling option for programmers (especially for ones doing scientific programming). So, when I started working through the [Data Structures and Algorithms Specialisation](https://www.coursera.org/specializations/data-structures-algorithms) offered by UC San Diego and HSE University, on Coursera, I thought it would be nice to code the algorithms in *swift* and then compare the performance and ease of use with other languages like *Python*, *C/C++* and *Java* (I'll not be coding in all these, but just use the benchmark provided in the course and try to match the *C* performance).

As I started reading about the language, apart from things like *Type Safety* and *Automatic Memory Management*, what stood out the most to me was the amazing [*Swift Package Manager*](https://swift.org/package-manager/). It made organising and testing the code so easy (and in a way automatic) that I was focusing more on the actual Algorithm and everything else was taken care of automatically.

In this post, 
- I'll be introducing *Swift Package Manager* and explain everything by working on a very simple problem of **Pairwise Product**. 
- I'll demonstrate how to we can create an *executable* so easily in the *swift programming language*, and then test different parts of the algorithm separately to isolate the error.

> Note: I'll be using Linux (most of the tutorials online use MacOS, soo this provides something different).

## Problem Statement

### Maximum Pairwise Product Problem
*Find the maximum product of two distinct numbers in a sequence of non-negative integers.*
- **Input**: A sequence of non-negative integers.
- **Output**: The maximum value that can be obtained by multiplying two different elements from the sequence.

1) **Input Format**: The first line contains an integer $$n$$. The next line contains $$n$$ non-negative integers $$a_1 , . . . , a_n$$ (separated by spaces).

2) **Output Format**: Maximum Pairwise Product

3) **Constraints**: 
    - Length of the sequence:  $$2 \le n \le 2 \cdot 10^5$$ 
    - Each element of the sequence:  $$0 \le a_i \le 2 \cdot 10^5$$

4) **Time limits**: 
    - *C/C++*: 1 sec
    - *Python*: 5 secs
    - *Java*: 1.5 secs

5) **Memory limits**: 512 Mb

## Algorithms
I'll be implementing two different algorithms that do the same thing:
- *Naive Implementation*: Compute each pairwise product and then pick the maximum.
- *Fast Implementation*: Pick the largest and second largest number and then return their product.

### *Swift Package Manager (SPM)*
Before implementing, let's create a structure that'll help us organise our code better. This is where *Swift Package Manager* comes in handy as it handles everything for us automatically.  

First create an empty folder (named *02_pairwiseProd* here) by running the command `mkdir 02_pairwiseProd`, and to let *Swift Package Manager* handle subdirectories run the command `swift package init --type executable`. This will create several sub-directories automatically and their structure will look something like 
```
- Sources
    |- 02_pairwiseProd
        |- main.swift

- Tests
    |- 02_pairwiseProdTests
        |- _2_pairwiseProdTests.swift

- Package.swift

- README.md 
```

I'll explain the purpose of each folder and file as we move on to the implementation of algorithms.

### *Naive Implementation*
To better organise the code, I created a new folder named *core*, which is inside *Sources*. Then, I placed the file *naive.swift* with code inside this new folder. Updated structure will look like this:
```
- Sources
    |- 02_pairwiseProd
    |   |- main.swift
    |
    |- core
        |- naive.swift

- Tests
    |- 02_pairwiseProdTests
        |- _2_pairwiseProdTests.swift

- Package.swift

- README.md 
```

The algorithm is fairly simple and is written as follows:
```swift
// function to calculate max pairwise product in an inefficient way
public func naivePairwiseProd(arr: [Int]) -> Int {
    // variable to store max product
    var prod: Int = 0
    // finding length of array
    let arrLen = arr.count
    // looping through elements to create each product pair
    for i in 0..<(arrLen-1) {
        for j in i+1..<arrLen {
            let newProd = arr[i]*arr[j]
            // checking for max prod
            if newProd>prod {
                prod = newProd
            }
        }
    }
    return prod
}
```

> Note: The function is declared `public` so that it can be accessed by the *main.swift* file which will be used to call our program and is located in a different directory.

### *Fast Implementation*
The file containing an efficient implementation is also put inside *core* folder and named *fast.swift*. 

```swift
// function to calculate max pairwise product in a fast way
public func fastPairwiseProd(arr: [Int]) -> Int {
    // variable to store largest number index
    var id1: Int = 0
    // variable to store 2nd largest number index
    var id2: Int = 0
    // Constant to store product
    let prod: Int
    // finding length of array
    let arrLen = arr.count
    
    // looping through elements to locate the largest number
    for i in 1..<(arrLen) {
        if arr[i] > arr[id1] {
            id1 = i
        } 
    }
    
    if id1 == 0 {
        id2 = 1
    }

    // looping through elements to locate the 2nd largest number
    for i in 1..<(arrLen) {
        if i != id1 && arr[i] > arr[id2] {
            id2 = i
        } 
    }

    prod = arr[id1]*arr[id2]

    return prod
}
```

Updated structure is as follows:
```
- Sources
    |- 02_pairwiseProd
    |   |- main.swift
    |
    |- core
        |- naive.swift
        |- fast.swift

- Tests
    |- 02_pairwiseProdTests
        |- _2_pairwiseProdTests.swift

- Package.swift

- README.md 
```

### *main.swift*
> Remember that we intend to create an executable (that's why we ran `swift package init --type executable` command at the start).

`swift run` is the simple command that runs the executable from the terminal, but there's a lot more going on under the hood. When we run this command, the *SPM* locates *main.swift* inside the *Sources/02_pairwiseProd* and executes it. Hence, we need to add all the code surrounding the algorithms here.

The first step is to add relevant dependencies

```swift
import Foundation
import CoreFoundation

// importing core module that contains algorithms
import core
```
> Note: Each folder inside *Sources* directory is considered a separate *module* and needs importing before using.

Apart from entering array manually, I also gave an option to just mention the size of the array and the elements will get randomly assigned. 

```swift
// prompting to either enter the array manually or not
print("Want to enter array manually? (y/n): ")
let choice: String? = readLine()

let arr: [Int]
let correctProd: Int

// enter array manually if 'y'
if choice! == "y" {
    // Accept array
    print("Enter the elements of array separated by a space: ")
    // readLine() returns optional string
    let input1: String? = readLine()
    // Converting the String? into [String?] separated by a single space
    let arrStr = input1!.components(separatedBy: [" "])
    // Converting each element of [String?] to Int and storing in [Int]
    arr = arrStr.map{Int($0)!}
    // get length of array
    let arrLen = arr.count
} else {
    // Generate random array
    print("Enter the length of array: ", terminator: " ")
    // readLine() returns optional string
    let input2: String? = readLine() // reading length of array
    // randomly initialising array with elements between 0 and 200000
    arr = (0..<Int(input2!)!).map{_ in Int.random(in: 1...200000)}
}
```

Now that we have the array, let's call the defined algorithms.

```swift
// Calculating naive max product
var startTime = CFAbsoluteTimeGetCurrent()
let prodNaive = naivePairwiseProd(arr: arr)
var timeElapsed = CFAbsoluteTimeGetCurrent() - startTime
print("(NAIVE) Max pairwise product of input array: \(prodNaive)")
print("(NAIVE) Time elapsed: \(timeElapsed) s.")

// Calculating fast max product
startTime = CFAbsoluteTimeGetCurrent()
let prodFast = fastPairwiseProd(arr: arr)
let timeElapsed_fast = CFAbsoluteTimeGetCurrent() - startTime
print("(FAST) Max pairwise product of input array: \(prodFast)")
print("(FAST) Time elapsed: \(timeElapsed_fast) s.")

// Ratio of two speeds
print("NAIVE / FAST: \(timeElapsed/timeElapsed_fast)")
```

## Results
Before calling the executable, we need to compile our code files and then link them. Fortunately, *compiling*, *linking* and *calling* is all done by a single command: `swift run`. 

For *compiling* and *linking* to go smoothly, we need to tell the compiler to look for functions `naivePairwiseProd` and `fastPairwiseProd` inside *core* folder. This is done by adding contents of this folder as a dependency inside *Package.swift* file.

```swift
// swift-tools-version:5.3
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "02_pairwiseProd",
    dependencies: [
        // Dependencies declare other packages that this package depends on.
        // .package(url: /* package url */, from: "1.0.0"),
    ],
    targets: [
        // Targets are the basic building blocks of a package. A target can define a module or a test suite.
        // Targets can depend on other targets in this package, and on products in packages this package depends on.
        .target(
            name: "02_pairwiseProd",
            dependencies: ["core"]), // adding core dependencies
        .target(name: "core"), // can use import core now
        .testTarget(
            name: "02_pairwiseProdTests",
            dependencies: ["02_pairwiseProd", "core"]), // adding core dependencies for unit tests
    ]
)
```

> Note that I've mentioned `"core"` under dependencies in `target` and `testTarget`. This will ensure that compiler finds the contents of this folder while running *main.swift* and *_2_pairwiseProdTests.swift* (explained in next section).

Let's now run the command `swift run`.

Examples:

1) Manually entering arrays
    ```
    ❯ swift run
    [5/5] Build complete!
    Want to enter array manually? (y/n): 
    y
    Enter the elements of array separated by a space: 
    3 2 1 5 6 7 8 3 3
    (NAIVE) Max pairwise product of input array: 56
    (NAIVE) Time elapsed: 4.494190216064453e-05 s.
    (FAST) Max pairwise product of input array: 56
    (FAST) Time elapsed: 8.940696716308594e-06 s.
    NAIVE / FAST: 5.026666666666666
    ```
2) Random Arrays
    ```
    ❯ swift run
    [0/0] Build complete!
    Want to enter array manually? (y/n): 
    n
    Enter the length of array:  200000
    (NAIVE) Max pairwise product of input array: 39999600000
    (NAIVE) Time elapsed: 268.38322699069977 s.
    (FAST) Max pairwise product of input array: 39999600000
    (FAST) Time elapsed: 0.00775599479675293 s.
    NAIVE / FAST: 34603.327364667544
    ```

> Notice that for longer arrays, *fast implementation* is $$\approx 34000 \times$$ faster. That's a big number!!!

Everthing looks good, but hang on. How can we be sure that the output is actually correct?

This is where **Unit Testing** comes in and *SPM* has made this very simple.

## Unit Testing
To add tests for the algorithm, just add functions inside the `final class` body. For example, I wrote three tests:
1) Tests *Naive Implementation* against a standard algorithm (that I know is corect).
2) Tests *Fast Implementation* against a standard algorithm.
3) Compares *Naive Implementation* and *Fast Implementation*.

```swift
import XCTest
import class Foundation.Bundle

import core

final class _2_max_pairwise_prodTests: XCTestCase {
    // Testing Naive against standard
    func testNaive() throws {
      var i: Int = 0
      while i < 10000 {
        let input2: Int = Int.random(in: 2...2000)
        let arr: [Int] = (0..<input2).map{_ in Int.random(in: 0...200000)}
        let arrSorted = arr.sorted()
        let correctProd: Int = arrSorted[input2-1]*arrSorted[input2-2]

        // checking naive prod
        let prodNaive: Int = naivePairwiseProd(arr: arr)
        XCTAssertEqual(prodNaive, correctProd)
        
        i += 1
      }
    }

    // Testing Fast against standard
    func testFast() throws {
      var i: Int = 0
      while i < 10000 {
        let input2: Int = Int.random(in: 2...2000)
        let arr: [Int] = (0..<input2).map{_ in Int.random(in: 0...200000)}
        let arrSorted = arr.sorted()
        let correctProd: Int = arrSorted[input2-1]*arrSorted[input2-2]

        // checking fast prod
        let prodFast: Int = fastPairwiseProd(arr: arr)
        XCTAssertEqual(prodFast, correctProd)
        
        i += 1
      }
    }

    // Comparing naive and fast
    func testCompare() throws {
      var i: Int = 0
      while i < 10000 {
        let input2: Int = Int.random(in: 2...2000)
        let arr: [Int] = (0..<input2).map{_ in Int.random(in: 0...200000)}

        // naive prod
        let prodNaive: Int = naivePairwiseProd(arr: arr)

        // fast prod
        let prodFast: Int = fastPairwiseProd(arr: arr)
      
        XCTAssertEqual(prodFast, prodNaive)
        
        i += 1
      }
    }
}
```

> Note: Functions here have to start with work `test`.

Now we have 2 different options here:
- Run each test differently; `swift test --filter funcName`
    
    For example, testing *Fast Implementation*:
    ```
    ❯ swift test --filter testFast
    [8/8] Build complete!
    Test Suite 'Selected tests' started at 2021-07-09 18:54:30.327
    Test Suite '_2_max_pairwise_prodTests' started at 2021-07-09 18:54:30.328
    Test Case '_2_max_pairwise_prodTests.testFast' started at 2021-07-09 18:54:30.328
    Test Case '_2_max_pairwise_prodTests.testFast' passed (38.936 seconds)
    Test Suite '_2_max_pairwise_prodTests' passed at 2021-07-09 18:55:09.265
            Executed 1 test, with 0 failures (0 unexpected) in 38.936 (38.936) seconds
    Test Suite 'Selected tests' passed at 2021-07-09 18:55:09.265
            Executed 1 test, with 0 failures (0 unexpected) in 38.936 (38.936) seconds
    ```
- Run everything all together; `swift test`:
    ```
    ❯ swift test
    [0/0] Build complete!
    Test Suite 'All tests' started at 2021-07-09 18:55:52.685
    Test Suite 'debug.xctest' started at 2021-07-09 18:55:52.686
    Test Suite '_2_max_pairwise_prodTests' started at 2021-07-09 18:55:52.686
    Test Case '_2_max_pairwise_prodTests.testCompare' started at 2021-07-09 18:55:52.686
    Test Case '_2_max_pairwise_prodTests.testCompare' passed (120.03 seconds)
    Test Case '_2_max_pairwise_prodTests.testFast' started at 2021-07-09 18:57:52.716
    Test Case '_2_max_pairwise_prodTests.testFast' passed (43.093 seconds)
    Test Case '_2_max_pairwise_prodTests.testNaive' started at 2021-07-09 18:58:35.808
    Test Case '_2_max_pairwise_prodTests.testNaive' passed (132.429 seconds)
    Test Suite '_2_max_pairwise_prodTests' passed at 2021-07-09 19:00:48.238
            Executed 3 tests, with 0 failures (0 unexpected) in 295.552 (295.552) seconds
    Test Suite 'debug.xctest' passed at 2021-07-09 19:00:48.238
            Executed 3 tests, with 0 failures (0 unexpected) in 295.552 (295.552) seconds
    Test Suite 'All tests' passed at 2021-07-09 19:00:48.238
            Executed 3 tests, with 0 failures (0 unexpected) in 295.552 (295.552) seconds
    ```

## Conclusions
- Our *fast implementation* passed the *time* and *memory* limits by executing on the array of size 200000 under 1 sec. 
- The *swift programming language* is fast enough to compete with benchmarks set for *C/C++* while being *Python* like expressive.
- *SPM* is an amazing utility that has made *organising* and *unit testing* the code super easy.

## References
- [Data Structures and Algorithms Specialization](https://www.coursera.org/specializations/data-structures-algorithms)
- [Github link for my complete code](https://github.com/tgautam03/DataStructures-Algorithms/tree/main/UC-San-Diego/C1-Algo-Toolbox/W1/02_pairwiseProd)
- [Swift Package Manager](https://swift.org/package-manager/)