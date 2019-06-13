#ifndef _CONC_LINKED_LIST_
#define _CONC_LINKED_LIST_

//TODO: wrap device API calls in an error handler (already wrote wrapper for host API calls)

/*  */
template <class T>
class DeviceLinkedList
{
private:
	struct Node
	{
		int index;
		T data;
		Node* next;
	};

	/* variables that hold info about the linked list */
	int size;
	Node* start;
	Node* end;

public:
	/* constructor */
	ConcLinkedList()
	{
		cudaMalloc(start, sizeof(Node));
		start->index = 0;
		start->data = NULL;
		start->next = NULL;
		end = start;
		size = 0;
	}

	/* deconstructor */
	~ConcLinkedList()
	{
		Node* temp = start;
		Node* toFree;

		while(temp != null)
		{
			toFree = temp;
			temp = temp->next;
			cudaFree(toFree);
		}
	}
	
	/* add value to end of list */
	void add(T* value)
	{
		if (size > 0)
		{
			cudaMalloc(end->next, sizeof(Node));
			end = end->next;
			end->value = *value;
			size++;
		}
		else
		{
			start->data = *value;
			size++;
		}
	}

	/* get value in node at index i */
	T get(int i)
	{
		if (i > size)
			return NULL;

		Node* temp = start;
		while (temp->index < i)
			temp = temp->next;
		return *(temp->data);
	}

	/* remove node at index i */
	int remove(int i)
	{
		if (i > size)
			return -1;

		Node* prev = start;
		while (prev->index < i-1)
			prev = prev->next;

		Node* toFree = prev->next;
		prev->next = toFree->next;
		cudaFree(toFree);
		return 1;
	}
};


#endif